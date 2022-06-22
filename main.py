import torch
from torch.utils.data import DataLoader
from config import *
from dataset.dataset import *
from statistic.collect_stat import CollectStatistics
from util.util import split_data, NodeSampler
import numpy as np
import random
import copy
from model.model import Model
from util.util import DatasetSplit

if device.type != 'cpu':
    torch.cuda.set_device(device)

if __name__ == "__main__":
    stat = CollectStatistics(results_file_name=results_file)

    for seed in simulations:

        random.seed(seed)
        np.random.seed(seed)  # numpy
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed(seed)  # gpu
        torch.backends.cudnn.deterministic = True  # cudnn

        data_train, data_test = load_data(dataset, dataset_file_path, 'cpu')
        data_train_loader = DataLoader(data_train, batch_size=batch_size_eval, shuffle=True, num_workers=0)
        data_test_loader = DataLoader(data_test, batch_size=batch_size_eval, num_workers=0)
        dict_users = split_data(dataset, data_train, n_nodes)
        if n_nodes is None:
            n_nodes = len(dict_users)

        node_sampler = NodeSampler(n_nodes, permutation=use_permute)

        model = Model(seed, step_size, model_name=model_name, device=device, flatten_weight=True,
                      pretrained_model_file=load_model_file)

        train_loader_list = []
        dataiter_list = []
        for n in range(n_nodes):
            train_loader_list.append(
                DataLoader(DatasetSplit(data_train, dict_users[n]), batch_size=batch_size_train, shuffle=True))
            dataiter_list.append(iter(train_loader_list[n]))


        def sample_minibatch(n):
            try:
                images, labels = dataiter_list[n].next()
                if len(images) < batch_size_train:
                    dataiter_list[n] = iter(train_loader_list[n])
                    images, labels = dataiter_list[n].next()
            except StopIteration:
                dataiter_list[n] = iter(train_loader_list[n])
                images, labels = dataiter_list[n].next()

            return images, labels

        def sample_full_batch(n):
            images = []
            labels = []
            for i in range(len(train_loader_list[n].dataset)):
                images.append(train_loader_list[n].dataset[i][0])

                l = train_loader_list[n].dataset[i][1]
                if not isinstance(l, torch.Tensor):
                    l = torch.as_tensor(l)
                labels.append(l)

            return torch.stack(images), torch.stack(labels)

        w_global = model.get_weight()   # Get initial weight

        num_iter = 0
        last_output = 0
        last_amplify = 0
        last_save_latest = 0
        last_save_checkpoint = 0

        while True:
            print('seed', seed,'  iteration', num_iter)

            node_sample_set = [n for n in range(0, n_nodes)]
            node_subset = node_sampler.sample(node_sample_set, n_nodes_in_each_round)

            accumulated = 0

            for n in node_subset:
                model.assign_weight(w_global)
                model.model.train()

                for i in range(0, iters_per_round):
                    images, labels = sample_minibatch(n)

                    images, labels = images.to(device), labels.to(device)

                    if transform_train is not None:
                        images = transform_train(images)

                    model.optimizer.zero_grad()
                    output = model.model(images)
                    loss = model.loss_fn(output, labels)
                    loss.backward()
                    model.optimizer.step()

                w_tmp = model.get_weight()  # deepcopy is already included here
                w_tmp -= w_global

                if accumulated == 0:  # accumulated weights
                    w_accumulate = w_tmp
                    # Note: w_tmp cannot be used after this
                else:
                    w_accumulate += w_tmp

                accumulated += 1

            if accumulated > 0:
                w_global += torch.div(w_accumulate, torch.tensor(accumulated).to(device)).view(-1)

            num_iter = num_iter + iters_per_round

            if save_checkpoint and num_iter - last_save_checkpoint >= iters_checkpoint:
                torch.save(model.model.state_dict(), save_model_file + '-checkpoint-sim-' + str(seed) + '-iter-' + str(num_iter))
                last_save_checkpoint = num_iter

            if num_iter - last_output >= min_iters_per_eval:
                stat.collect_stat(seed, num_iter, model, data_train_loader, data_test_loader, w_global)
                last_output = num_iter

            if num_iter >= max_iter:
                break

        del model
        del w_global
        del w_accumulate

        torch.cuda.empty_cache()

