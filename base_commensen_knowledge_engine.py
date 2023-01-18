import math
import sys
from typing import Dict, Iterable, Optional

import torch
import torch.nn
import torch.optim

import util.dist as dist
from datasets.vidstg_eval import VidSTGEvaluator
from datasets.hcstvg_eval import HCSTVGEvaluator
from util.metrics import MetricLogger, SmoothedValue
from util.misc import targets_to
from util.optim import adjust_learning_rate, update_ema


def train_one_epoch(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    max_norm: float = 0,
    model_ema: Optional[torch.nn.Module] = None,
    writer=None,
):
    model.train()
    if criterion is not None:
        criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "lr_text_encoder", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100

    num_training_steps = int(len(data_loader) * args.epochs)
    for i, batch_dict in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        curr_step = epoch * len(data_loader) + i
        samples = batch_dict["samples"].to(device)
        if "samples_fast" in batch_dict:
            samples_fast = batch_dict["samples_fast"].to(device)
        else:
            samples_fast = None
        durations = batch_dict["durations"]
        captions = batch_dict["captions"]
        targets = batch_dict["targets"]

        targets = targets_to(targets, device)

        # forward
        memory_cache = model(
            samples,
            durations,
            captions,
            encode_and_save=True,
            samples_fast=samples_fast,
        )
        outputs = model(
            samples,
            durations,
            captions,
            encode_and_save=False,
            memory_cache=memory_cache,
        )

        # only keep box predictions in the annotated moment
        max_duration = max(durations)
        device = outputs["pred_boxes"].device
        inter_idx = batch_dict["inter_idx"]
        keep_list = []
        for i_dur, (duration, inter) in enumerate(zip(durations, inter_idx)):
            keep_list.extend(
                [
                    elt
                    for elt in range(
                        i_dur * max_duration + inter[0],
                        (i_dur * max_duration) + inter[1] + 1,
                    )
                ]
            )
        keep = torch.tensor(keep_list).long().to(device)
        outputs["pred_boxes"] = outputs["pred_boxes"][keep]
        for i_aux in range(len(outputs["aux_outputs"])):
            outputs["aux_outputs"][i_aux]["pred_boxes"] = outputs["aux_outputs"][i_aux][
                "pred_boxes"
            ][keep]
        b = len(durations)
        targets = [
            x for x in targets if len(x["boxes"])
        ]  # keep only targets in the annotated moment
        assert len(targets) == len(outputs["pred_boxes"]), (
            len(outputs["pred_boxes"]),
            len(targets),
        )
        # mask with padded positions set to False for loss computation
        if args.sted:
            time_mask = torch.zeros(b, outputs["pred_sted"].shape[1]).bool().to(device)
            for i_dur, duration in enumerate(durations):
                time_mask[i_dur, :duration] = True
        else:
            time_mask = None

        # compute losses
        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, inter_idx, time_mask))

        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        adjust_learning_rate(
            optimizer,
            epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )
        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        if writer is not None and dist.is_main_process() and i % 100 == 0:
            for k in loss_dict_reduced_unscaled:
                writer.add_scalar(f"{k}", metric_logger.meters[k].avg, i)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def dijkstra(video_action_str, sentence_action_str, middle_action_map):
    total_action_list = list(middle_action_map.keys())
    for key in middle_action_map.keys():
        key_action_list = middle_action_map[key]
        for action_item in key_action_list:
            if action_item not in total_action_list:
                total_action_list.append(action_item)
    distance = {}
    for action_item in total_action_list:
        distance[action_item] = float('inf')
    distance[video_action_str] = 0
    V = len(total_action_list)
    used = [False for _ in range(V)]
    last_node = {}
    for action_item in total_action_list:
        last_node[action_item] = action_item
    while True:
        v=-1
        for u in range(V):
            if not used[u] and \
                (v==-1 or distance[total_action_list[u]]<distance[total_action_list[v]]):
                v=u
        if v==-1:
            break
        used[v] = True
        for u in range(V):
            u_str = total_action_list[u]
            v_str = total_action_list[v]
            if v_str not in middle_action_map.keys():
                cost_v_u = float('inf')
            elif u_str not in middle_action_map[v_str]:
                cost_v_u = float('inf')
            else:
                cost_v_u = 1
            # print("1111111111111111111")
            # print(distance[u_str])
            # print(distance[v_str]+cost_v_u)
            if distance[u_str] > distance[v_str]+cost_v_u:
                last_node[u_str] = v_str
                distance[u_str] = distance[v_str]+cost_v_u
    middle_action_list = []
    if sentence_action_str is not None:
        tem_action_str = sentence_action_str
        while True:
            tem_action_str = last_node[tem_action_str]
            if tem_action_str == video_action_str:
                break
            else:
                middle_action_list.append(tem_action_str)
    return middle_action_list, distance

def middle_action_pred(outputs, video_action_list, middle_action_map,
                       sentence_action_str=None, candidated_action_index=None):
    video_action_prob = outputs['pred_action'][0]
    # _, video_action_index = torch.max(video_action_prob, dim=-1)

    # print("333333333333333333333333333")
    # print(video_action_prob.size())
    # print(len(video_action_list))
    # print(video_action_list)

####################如果预测文本
    if len(candidated_action_index) == 1:
        video_action_index_int = candidated_action_index[0]
        video_action_str = video_action_list[video_action_index_int]
        return [], video_action_str

    if len(candidated_action_index) == 0:
        _, video_action_index = torch.max(video_action_prob, dim=-1)
        video_action_index_int = video_action_index.cpu().item()
        video_action_str = video_action_list[video_action_index_int]
        # print("00000000000000000000000000000000")
        # print(video_action_str)
        # exit(0)
        return [], video_action_str

    max_index = candidated_action_index[0]
    for index in candidated_action_index:
#############后面+1后再使用
        video_action_prob_max = video_action_prob[max_index].cpu().item()
        video_action_prob_tem = video_action_prob[index].cpu().item()
#############
        if video_action_prob_tem>video_action_prob_max:
            max_index = index
    video_action_index_int = max_index#video_action_index.cpu().item()-1
    video_action_str = video_action_list[video_action_index_int]

    # print("1111111111111111111")
    # print(candidated_action_index)
    # print(video_action_str)
    # print(sentence_action_str)
    # print(video_action_list)
    # print(middle_action_map)

    middle_action_list, distance = dijkstra(video_action_str,
                                            sentence_action_str, middle_action_map)
    # print("2222222222222222222222")
    return middle_action_list, video_action_str

def reverse_middle_action(middle_action_map):
    reverse_middle_action_map = {}
    total_action_list = list(middle_action_map.keys())
    for key in middle_action_map.keys():
        key_action_list = middle_action_map[key]
        for action_item in key_action_list:
            if action_item not in total_action_list:
                total_action_list.append(action_item)
    for action in total_action_list:
        for key in middle_action_map.keys():
            if action in middle_action_map[key]:
                if action not in reverse_middle_action_map.keys():
                    reverse_middle_action_map[action] = [key]
                else:
                    reverse_middle_action_map[action].append(key)
    return reverse_middle_action_map

def get_candidated_action(video_action_list,
                    middle_action_map, sentence_action_str):
    reverse_middle_action_map = reverse_middle_action(middle_action_map)
    _, distance = dijkstra(sentence_action_str,
                                None, reverse_middle_action_map)
    candidated_action_list = []
    distance_limit = 9999

################修改成文本中的动作也视为候选节点
    for key in distance.keys():
        if distance[key]<distance_limit and distance[key]>=0 and key in video_action_list:
            candidated_action_list.append(video_action_list.index(key))



    # print("22222222222222222222222")
    # # print(video_action_list)
    # print(candidated_action_list)
    # # print(sentence_action_str)
    # print(reverse_middle_action_map)
    # # print(middle_action_map)

    return candidated_action_list

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    postprocessors: Dict[str, torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader,
    evaluator_list,
    device: torch.device,
    args,
):
    print("测试")
    model.eval()
    if criterion is not None:
        criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    count=0
    attribute = False
    #print("这里在读入数据之前")
    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, 100, header)
    ):
        samples = batch_dict["samples"].to(device)
        if "samples_fast" in batch_dict:
            samples_fast = batch_dict["samples_fast"].to(device)
        else:
            samples_fast = None
        durations = batch_dict["durations"]
        captions = batch_dict["captions"]
        targets = batch_dict["targets"]
        video_ids = batch_dict["video_ids"]
        video_action_str_label = batch_dict["video_action_str"]

        personAttr = batch_dict['personAttr']
        sentiment = batch_dict['sentiment']
        scene = batch_dict['scene']
        video_original_id = batch_dict['video_original_id']

####################假设句子中的动作已经写进了列表中
        action_list_str = batch_dict["action_list_str"]
        # print(action_list_str)
        sentence_action_str = action_list_str[0][-1]

        targets = targets_to(targets, device)

        # forward
        memory_cache = model(samples, durations, captions, encode_and_save=True, samples_fast=samples_fast)
        outputs = model(samples, durations, captions, encode_and_save=False, memory_cache=memory_cache)

###################产生动作预测结果
        ############反向动作序列，找到候选动作
        candidated_action_index = get_candidated_action(model.video_action_list,
                                        model.middle_action_map, sentence_action_str)

        # print("00000000000000000000000000000")
        video_action_str_list = []
        for index_item in candidated_action_index:
            # video_action_index_int = candidated_action_index[index_item]
            video_action_str = model.video_action_list[index_item]
            video_action_str_list.append(video_action_str)
        # print(video_action_str_list)
        # print(video_action_str_label)
        # print(sentence_action_str)

        middle_action_list, video_action_str = middle_action_pred(outputs, model.video_action_list,
                                model.middle_action_map, sentence_action_str, candidated_action_index)

        if video_action_str not in model.video_action_list:
            print(video_action_str)
            print(video_original_id)
            exit()
        else:
            action_target = torch.zeros(1)  # len(model.video_action_list)+1)
            index = model.video_action_list.index(video_action_str)
            index += 1  # 空出来None的位置
            action_target[0] = index
            action_loss_flag = bool(index)

        action_target = action_target.to(device)

        # only keep box predictions in the annotated moment
        max_duration = max(durations)
        inter_idx = batch_dict["inter_idx"]

        keep_list = []
        for i_dur, (duration, inter) in enumerate(zip(durations, inter_idx)):
            if inter[0] >= 0:
               keep_list.extend(
                   [elt for elt in range(i_dur * max_duration + inter[0],(i_dur * max_duration) + inter[1] + 1)]
                )
        keep = torch.tensor(keep_list).long().to(outputs["pred_boxes"].device)
        if args.test:
            pred_boxes_all = outputs["pred_boxes"]
            targets_all = [x for x in targets]

        outputs["pred_boxes"] = outputs["pred_boxes"][keep]
        for i_aux in range(len(outputs["aux_outputs"])):
            outputs["aux_outputs"][i_aux]["pred_boxes"] = outputs["aux_outputs"][i_aux]["pred_boxes"][keep]

        b = len(durations)
        targets = [x for x in targets if len(x["boxes"])]
        assert len(targets) == len(outputs["pred_boxes"]), (len(targets), len(outputs["pred_boxes"]))
        # mask with padded positions set to False for loss computation
        if args.sted:
            #print("!!!!!!basetrain_engine!!!!")
            time_mask = torch.zeros(b, outputs["pred_sted"].shape[1]).bool().to(device)
            for i_dur, duration in enumerate(durations):
                time_mask[i_dur, :duration] = True
        else:
            time_mask = None

            qtypes = batch_dict["qtype"]
            assert len(set(video_ids)) == len(qtypes)
            if args.sted:
                assert len(pred_steds) == len(qtypes)
                for video_id, pred_sted in zip(video_ids, pred_steds):
                    vidstg_video_res[video_id] = {"sted": pred_sted, "qtype": qtypes[video_id]}
            else:
                for video_id in video_ids:
                    vidstg_video_res[video_id] = {"qtype": qtypes[video_id]}
            res = {target["image_id"]: output for target, output in zip(targets, results)}

    metric_logger.synchronize_between_processes()
    # print("11111111111111111")
    #print("Averaged stats:", metric_logger) #和前面的bfore stats 数据时一样的
    for evaluator in evaluator_list:
        #print("aaaaaaaaaaaa")
        evaluator.synchronize_between_processes()

    vidstg_res = None
    hcstvg_res = None
    for evaluator in evaluator_list:
        if isinstance(evaluator, VidSTGEvaluator):#pan duan shi fou yi zhi lei xing
            #print("bbbbbbbbbbbbbbb")
            vidstg_res = evaluator.summarize()
            #print("**************")
        elif isinstance(evaluator, HCSTVGEvaluator):
            hcstvg_res = evaluator.summarize()    
        elif "hcstvg" in postprocessors.keys():
            #print("no use")
            video_ids = batch_dict["video_ids"]
            frames_id = batch_dict["frames_id"]
            if args.sted:
                pred_steds = postprocessors["hcstvg"](outputs, frames_id, video_ids=video_ids, time_mask=time_mask)
            image_ids = [t["image_id"] for t in targets]
            for im_id, result in zip(image_ids, results):
                hcstvg_res[im_id] = {"boxes": [result["boxes"].detach().cpu().tolist()]}


            if args.sted:
                assert len(set(video_ids)) == len(pred_steds)
                for video_id, pred_sted in zip(video_ids, pred_steds):
                    hcstvg_video_res[video_id] = {"sted": pred_sted}
            else:
                hcstvg_video_res[video_id] = {}
            res = {target["image_id"]: output for target, output in zip(targets, results)}
        else:
            res = {target["image_id"].item(): output for target, output in zip(targets, results)}


        for evaluator in evaluator_list:
            if isinstance(evaluator, VidSTGEvaluator):
                evaluator.update(vidstg_res)
                evaluator.video_update(vidstg_video_res)

                evaluator.action_update(video_ids[0], video_action_str, middle_action_list)

                if args.test:
                    tsa_weights = [outputs["aux_outputs"][i_aux]["weights"] for i_aux in range(len(outputs["aux_outputs"]))]
                    tsa_weights.append(outputs["weights"])
                    weights = torch.stack(tsa_weights)
                    ca_weights = [outputs["aux_outputs"][i_aux]["ca_weights"] for i_aux in range(len(outputs["aux_outputs"]))]
                    ca_weights.append(outputs["ca_weights"])
                    ca_weights = torch.stack(ca_weights)
                    text_weights = ca_weights[..., -len(memory_cache["text_memory_resized"]) :]
                    spatial_weights = ca_weights[..., : -len(memory_cache["text_memory_resized"])].reshape(
                        ca_weights.shape[0],
                        ca_weights.shape[1],
                        ca_weights.shape[2],
                        math.ceil(samples.tensors.shape[2] / 32),
                        -1,
                    )  # hw
                    # tokens = memory_cache['tokenized'].tokens()
                    evaluator.save(weights, text_weights, spatial_weights, outputs["pred_sted"], image_ids, video_ids)
            elif isinstance(evaluator, HCSTVGEvaluator):
                evaluator.update(hcstvg_res)
                evaluator.video_update(hcstvg_video_res)
            else:
                evaluator.update(res)

    metric_logger.synchronize_between_processes()
    # print("11111111111111111")
    #print("Averaged stats:", metric_logger) #和前面的bfore stats 数据时一样的
    for evaluator in evaluator_list:
        #print("aaaaaaaaaaaa")
        evaluator.synchronize_between_processes()

    vidstg_res = None
    hcstvg_res = None
    for evaluator in evaluator_list:
        if isinstance(evaluator, VidSTGEvaluator):#pan duan shi fou yi zhi lei xing
            #print("bbbbbbbbbbbbbbb")
            vidstg_res = evaluator.summarize()
            #print("**************")
        elif isinstance(evaluator, HCSTVGEvaluator):
            hcstvg_res = evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    #print("stats:",stats)
    # print("33333333333333333")

    if vidstg_res is not None:
        #print("cccccccccc ")
        stats["vidstg"] = vidstg_res

    if hcstvg_res is not None:
        stats["hcstvg"] = hcstvg_res
    #print("stats:",stats)
    return stats

