import os

from attrdict import AttrDict
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj
from experiments.graph_classification import Experiment

import time
import tqdm
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, digl, borf

mutag = list(TUDataset(root="data", name="MUTAG", use_node_attr=True))
enzymes = list(TUDataset(root="data", name="ENZYMES", use_node_attr=True))
proteins = list(TUDataset(root="data", name="PROTEINS", use_node_attr=True))
imdb = list(TUDataset(root="data", name="IMDB-BINARY", use_node_attr=True))
datasets = {"mutag": mutag, "enzymes": enzymes, "imdb": imdb, "proteins": proteins}
for key in datasets:
    if key in ["reddit", "imdb", "collab"]:
        for graph in datasets[key]:
            n = graph.num_nodes
            graph.x = torch.ones((n, 1))


def average_spectral_gap(dataset):
    # computes the average spectral gap out of all graphs in a dataset
    spectral_gaps = []
    for graph in dataset:
        G = to_networkx(graph, to_undirected=True)
        spectral_gap = rewiring.spectral_gap(G)
        spectral_gaps.append(spectral_gap)
    return sum(spectral_gaps) / len(spectral_gaps)


def log_to_file(message, filename="results"):
    filename = filename + '/graph_classification.txt'
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()


default_args = AttrDict({
    "dropout": 0.5,
    "num_layers": 4,  # previous 4
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "layer_type": "R-GCN",
    "display": True,
    "num_trials": 100,
    "eval_every": 1,
    "rewiring": "sdrf",
    "num_iterations": 10,
    "patience": 100,
    "output_dim": 2,
    "alpha": 0.1,
    "eps": 0.001,
    "dataset": None,
    "last_layer_fa": False,  # prev False
    "borf_batch_add": 4,
    "borf_batch_remove": 2,
    "sdrf_remove_edges": False,
    "results_save_dir": 'results',
})

hyperparams = {
    "mutag": AttrDict({"output_dim": 2}),
    "enzymes": AttrDict({"output_dim": 6}),
    "proteins": AttrDict({"output_dim": 2}),
    "collab": AttrDict({"output_dim": 3}),
    "imdb": AttrDict({"output_dim": 2}),
    "reddit": AttrDict({"output_dim": 2})
}

results = []
args = default_args
args += get_args_from_input()
if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}

for key in datasets:
    args += hyperparams[key]
    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []
    energies = []
    print(f"TESTING: {key} ({args.rewiring} - layer {args.layer_type})")
    dataset = datasets[key]

    print('REWIRING STARTED...')
    start = time.time()
    with tqdm.tqdm(total=len(dataset)) as pbar:
        if args.rewiring == "fosr":
            for i in range(len(dataset)):
                edge_index, edge_type, _ = fosr.edge_rewire(dataset[i].edge_index.numpy(),
                                                            num_iterations=args.num_iterations)
                dataset[i].edge_index = torch.tensor(edge_index)
                dataset[i].edge_type = torch.tensor(edge_type)
                pbar.update(1)
        elif args.rewiring == "sdrf_orc":
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = sdrf.sdrf(dataset[i], loops=args.num_iterations,
                                                                        remove_edges=False, is_undirected=True,
                                                                        curvature='orc')
                pbar.update(1)
        elif args.rewiring == "sdrf_bfc":
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = sdrf.sdrf(dataset[i], loops=args.num_iterations,
                                                                        remove_edges=args["sdrf_remove_edges"],
                                                                        is_undirected=True, curvature='bfc')
                pbar.update(1)
        elif args.rewiring == "borf":
            print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
            print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
            print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
            if args.use_edge_weight:
                print('using edge weight, not rewiring...')
                # print('Using min-max normalization for the final edge weight!!!')
                for i in range(len(dataset)):
                    dataset[i].edge_index, dataset[i].edge_type, dataset[i].edge_weight = \
                        borf.borf_edge_weight(dataset[i],
                                              loops=args.num_iterations,
                                              remove_edges=False,
                                              is_undirected=True,
                                              batch_add=args.borf_batch_add,
                                              batch_remove=args.borf_batch_remove,
                                              alpha=args.alpha,
                                              dataset_name=args.save_dataset_name,
                                              graph_index=i)

                    # dataset[i].edge_index, dataset[i].edge_type, dataset[i].edge_weight = \
                    #     borf.borf_edge_weight_minmax_normalization(dataset[i],
                    #                                                loops=args.num_iterations,
                    #                                                remove_edges=False,
                    #                                                is_undirected=True,
                    #                                                batch_add=args.borf_batch_add,
                    #                                                batch_remove=args.borf_batch_remove,
                    #                                                alpha=args.alpha,
                    #                                                dataset_name=args.save_dataset_name,
                    #                                                graph_index=i)
                    pbar.update(1)
            else:  # not using edge weight, rewiring graph instead
                for i in range(len(dataset)):
                    dataset[i].edge_index, dataset[i].edge_type = borf.borf3(dataset[i],
                                                                             loops=args.num_iterations,
                                                                             remove_edges=False,
                                                                             is_undirected=True,
                                                                             batch_add=args.borf_batch_add,
                                                                             batch_remove=args.borf_batch_remove,
                                                                             alpha=args.alpha,
                                                                             dataset_name=args.save_dataset_name,
                                                                             graph_index=i)
                    pbar.update(1)
        elif args.rewiring == "digl":
            for i in range(len(dataset)):
                dataset[i].edge_index = digl.rewire(dataset[i], alpha=0.1, eps=0.05)
                m = dataset[i].edge_index.shape[1]
                dataset[i].edge_type = torch.tensor(np.zeros(m, dtype=np.int64))
                pbar.update(1)
    end = time.time()
    rewiring_duration = end - start

    # spectral_gap = average_spectral_gap(dataset)
    print('TRAINING STARTED...')
    start = time.time()
    for trial in range(args.num_trials):
        train_acc, validation_acc, test_acc, energy = Experiment(args=args, dataset=dataset).run()
        # TODO: comment: seems inside the run loop, even for GCNConv which can take edge weight into consideration,
        #  the code didn't include it (e.g., pass "edge_weight" in the forward pass)
        train_accuracies.append(train_acc)
        validation_accuracies.append(validation_acc)
        test_accuracies.append(test_acc)
        energies.append(energy)
    end = time.time()
    run_duration = end - start

    train_mean = 100 * np.mean(train_accuracies)
    val_mean = 100 * np.mean(validation_accuracies)
    test_mean = 100 * np.mean(test_accuracies)
    energy_mean = 100 * np.mean(energies)

    train_std = np.std(train_accuracies)
    val_std = np.std(validation_accuracies)
    test_std = np.std(test_accuracies)
    energy_std = np.std(energies)

    train_ci = 2 * np.std(train_accuracies) / (args.num_trials ** 0.5)
    val_ci = 2 * np.std(validation_accuracies) / (args.num_trials ** 0.5)
    test_ci = 2 * np.std(test_accuracies) / (args.num_trials ** 0.5)
    energy_ci = 200 * np.std(energies) / (args.num_trials ** 0.5)
    import os

    args.results_save_dir = args.results_save_dir + '/' + args.save_dataset_name
    if not os.path.exists(args.results_save_dir):
        os.makedirs(args.results_save_dir)
    log_to_file(f"RESULTS FOR {key} ({args.rewiring}), {args.num_iterations} ITERATIONS:\n",
                filename=args.results_save_dir)
    log_to_file(f"average acc: {test_mean}\n", filename=args.results_save_dir)
    log_to_file(f"plus/minus:  {test_ci}\n\n", filename=args.results_save_dir)
    results.append({
        "dataset": key,
        "rewiring": args.rewiring,
        "layer_type": args.layer_type,
        "num_iterations": args.num_iterations,
        "borf_batch_add": args.borf_batch_add,
        "borf_batch_remove": args.borf_batch_remove,
        "sdrf_remove_edges": args.sdrf_remove_edges,
        "alpha": args.alpha,
        "eps": args.eps,
        "test_mean": test_mean,
        "test_ci": test_ci,
        "test_std": test_std,
        "val_mean": val_mean,
        "val_ci": val_ci,
        "val_std": val_std,
        "train_mean": train_mean,
        "train_ci": train_ci,
        "train_std": train_std,
        "energy_mean": energy_mean,
        "energy_ci": energy_ci,
        "last_layer_fa": args.last_layer_fa,
        "rewiring_duration": rewiring_duration,
        "run_duration": run_duration,
    })
    print(len(train_accuracies), len(validation_accuracies), len(test_accuracies))
    print(test_accuracies)

    # Log every time a dataset is completed
    df = pd.DataFrame(results)
    with open(args.results_save_dir + '/graph_classification_{}_{}.csv'.format(args.layer_type, args.rewiring),
              'a') as f:
        df.to_csv(f, mode='a', header=f.tell() == 0)
