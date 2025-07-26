import numpy as np
import torch
import torch.nn.functional as F
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
from qm9.analyze import check_stability
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion 

from types import SimpleNamespace
from qm9.models import get_optim

from utils import Queue

from qm9.rdkit_functions import BasicMolecularMetrics

def rotate_chain(z):
    assert z.size(0) == 1

    z_h = z[:, :, 3:]

    n_steps = 30
    theta = 0.6 * np.pi / n_steps
    Qz = torch.tensor(
        [[np.cos(theta), -np.sin(theta), 0.],
         [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    ).float()
    Qx = torch.tensor(
        [[1., 0., 0.],
         [0., np.cos(theta), -np.sin(theta)],
         [0., np.sin(theta), np.cos(theta)]]
    ).float()
    Qy = torch.tensor(
        [[np.cos(theta), 0., np.sin(theta)],
         [0., 1., 0.],
         [-np.sin(theta), 0., np.cos(theta)]]
    ).float()

    Q = torch.mm(torch.mm(Qz, Qx), Qy)

    Q = Q.to(z.device)

    results = []
    results.append(z)
    for i in range(n_steps):
        z_x = results[-1][:, :, :3]
        # print(z_x.size(), Q.size())
        new_x = torch.matmul(z_x.view(-1, 3), Q.T).view(1, -1, 3)
        # print(new_x.size())
        new_z = torch.cat([new_x, z_h], dim=2)
        results.append(new_z)

    results = torch.cat(results, dim=0)
    return results


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def finetune_ppo(args, device, flow: EnVariationalDiffusion, nodes_dist, dataset_info, wandb_logger=None):

    if args.dataset == 'qm9' or args.dataset == 'qm9_second_half' or args.dataset == 'qm9_first_half':
        n_nodes = 19
    elif args.dataset == 'geom':
        n_nodes = 44
    else:
        raise ValueError()

    ppo_config = SimpleNamespace()
    ppo_config.clip_range = 0.1
    ppo_config.max_grad_norm = 0.5
    ppo_config.inference_interval = 1
    ppo_config.n_samples = 4
    assert flow.T == 1000
    ppo_config.batch_size = (flow.T // ppo_config.inference_interval) // 10
    ppo_config.sample_n_nodes = True

    # node_mask = torch.ones(ppo_config.n_samples, n_nodes, 1).to(device)
    # edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    # edge_mask = edge_mask.repeat(ppo_config.n_samples, 1, 1).view(-1, 1).to(device)

    optim = get_optim(args, flow)
    gradnorm_queue = Queue()
    gradnorm_queue.add(3000)

    for i in range(100):
        if ppo_config.sample_n_nodes:
            nodesxsample = nodes_dist.sample(ppo_config.n_samples)
            # sample from the distribution of nodes
            # n_nodes = nodesxsample[0]
        else:
            nodesxsample = torch.tensor([n_nodes] * ppo_config.n_samples)

        max_n_nodes = dataset_info['max_n_nodes'] 
        # assert int(torch.max(nodesxsample)) <= max_n_nodes
        n_samples = ppo_config.n_samples
        # n_samples = len(nodesxsample)

        #[create node_mask with all ones]
        # node_mask = torch.ones(n_samples, n_nodes, 1).to(device)
        # edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
        # edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)

        #[create node_mask with max_n_nodes, allowing zero values]
        node_mask = torch.zeros(n_samples, max_n_nodes)
        for i in range(n_samples):
            node_mask[i, 0:nodesxsample[i]] = 1

        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(n_samples * max_n_nodes * max_n_nodes, 1).to(device)
        node_mask = node_mask.unsqueeze(2).to(device)

        metrics = flow.sample_chain_rl(node_mask=node_mask, edge_mask=edge_mask, context=None, ppo_config=ppo_config, device=device, optimizer=optim, gradnorm_queue=gradnorm_queue, dataset_info=dataset_info, nodesxsample=nodesxsample)
        if metrics is None:
            print('No metrics returned, skipping iteration.')
            continue
        metrics['General/timesteps_done'] = i * ppo_config.n_samples * flow.T
        metrics['General/episodes_done'] = i * ppo_config.n_samples
        if wandb_logger is not None:
            wandb_logger.log_and_dump(metrics)
        
        # print(f'Finetune PPO iteration {i}, avg_rewards: {metrics["rewards"]:.3f}, avg_advantages: {metrics["advantages"]:.3f}')

        # print(f'Finetune PPO iteration {i}')


def sample_chain(args, device, flow, n_tries, dataset_info, prop_dist=None):
    n_samples = 4
    if args.dataset == 'qm9' or args.dataset == 'qm9_second_half' or args.dataset == 'qm9_first_half':
        n_nodes = 19
    elif args.dataset == 'geom':
        n_nodes = 44
    else:
        raise ValueError()

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        context = prop_dist.sample(n_nodes).unsqueeze(1).unsqueeze(0)
        context = context.repeat(1, n_nodes, 1).to(device)
        #context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
    else:
        context = None

    node_mask = torch.ones(n_samples, n_nodes, 1).to(device)

    edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)

    if args.probabilistic_model == 'diffusion':
        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = flow.sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=100, dataset_info=dataset_info)
            chain = reverse_tensor(chain)

            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            x = chain[-1:, :, 0:3]
            one_hot = chain[-1:, :, 3:-1]
            one_hot = torch.argmax(one_hot, dim=2)

            atom_type = one_hot.squeeze(0).cpu().detach().numpy()
            x_squeeze = x.squeeze(0).cpu().detach().numpy()
            mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

            # Prepare entire chain.
            x = chain[:, :, 0:3]
            one_hot = chain[:, :, 3:-1]
            one_hot = F.one_hot(torch.argmax(one_hot, dim=2), num_classes=len(dataset_info['atom_decoder']))
            charges = torch.round(chain[:, :, -1:]).long()

            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')

    else:
        raise ValueError

    return one_hot, charges, x


def sample(args, device, generative_model, dataset_info,
           prop_dist=None, nodesxsample=torch.tensor([10]), context=None,
           fix_noise=False):
    max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0:nodesxsample[i]] = 1

    # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        if context is None:
            context = prop_dist.sample_batch(nodesxsample)
        context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
    else:
        context = None

    if args.probabilistic_model == 'diffusion':
        x, h = generative_model.sample(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise)

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        one_hot = h['categorical']
        charges = h['integer']

        assert_correctly_masked(one_hot.float(), node_mask)
        if args.include_charges:
            assert_correctly_masked(charges.float(), node_mask)

        rewards = generative_model.rewards_given_xh(x, one_hot, node_mask, dataset_info)
        # convert rewards to tensor
        rewards = torch.tensor(rewards, device=device)
        rate_positive = rewards > 0
        print(f"Rewards: {rewards}")
        print(f"Rate of positive rewards: {rate_positive.float().mean().item()}")

    else:
        raise ValueError(args.probabilistic_model)

    return one_hot, charges, x, node_mask


def sample_sweep_conditional(args, device, generative_model, dataset_info, prop_dist, n_nodes=19, n_frames=100):
    nodesxsample = torch.tensor([n_nodes] * n_frames)

    context = []
    for key in prop_dist.distributions:
        min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
        mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
        min_val = (min_val - mean) / (mad)
        max_val = (max_val - mean) / (mad)
        context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)
        context.append(context_row)
    context = torch.cat(context, dim=1).float().to(device)

    one_hot, charges, x, node_mask = sample(args, device, generative_model, dataset_info, prop_dist, nodesxsample=nodesxsample, context=context, fix_noise=True)
    return one_hot, charges, x, node_mask