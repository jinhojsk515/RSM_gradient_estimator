import torch
import math
import matplotlib.pyplot as plt
import numpy as np

import seaborn
from tqdm import tqdm
from value_gradient_estimate import Model, DiffusionProcess, GaussianDataset, reward_function



def main(args):
    mode = args.mode

    beta_1 = 1e-4
    beta_T = 0.03
    T = 500
    shape = (2,)


    device = torch.device('cuda')
    model = Model(device, beta_1, beta_T, T, shape[0])
    process = DiffusionProcess(beta_1, beta_T, T, model, device, shape=shape)

    minibatch_size = 2 ** 16
    model.load_state_dict(torch.load('./model_ft.pth'))
    model.eval()

    model_ref = Model(device, beta_1, beta_T, T, shape[0])
    model_ref.load_state_dict(torch.load('./model_base.pth'))
    model_ref.eval()

    dist1, dist2, dist3 = (3 - 0.5, -3 ** 0.5, 1), (-3 - 0.5, -3 ** 0.5, 1), (0 - 0.5, 2 * 3 ** 0.5, 1)
    tilted_dataset = GaussianDataset(dist1, dist2, dist3, probability=(1 / (math.exp(3) + math.exp(1.5) + 1), 1 - math.exp(1.5) / (math.exp(3) + math.exp(1.5) + 1)), total_len=1000)

    def opt_density(x, t):
        alpha = process.alpha_bars[t] ** 0.5
        return (torch.exp(torch.sum(-(x - torch.tensor([(3 - 0.5) * alpha, (-3 ** 0.5 - .0) * alpha], device=x.device)[None, :]) ** 2, dim=-1) / (2 * 1 ** 2)) * math.exp(-1.5) +
                torch.exp(torch.sum(-(x - torch.tensor([(-3 - 0.5) * alpha, (-3 ** 0.5 - .0) * alpha], device=x.device)[None, :]) ** 2, dim=-1) / (2 * 1 ** 2)) * math.exp(1.5) +
                torch.exp(torch.sum(-(x - torch.tensor([(0 - 0.5) * alpha, (2 * 3 ** 0.5 - .0) * alpha], device=x.device)[None, :]) ** 2, dim=-1) / (2 * 1 ** 2))) / (
                    1 + math.exp(-1.5) + math.exp(1.5))

    def ref_density(x, t):
        alpha = process.alpha_bars[t] ** 0.5
        return (torch.exp(torch.sum(-(x - torch.tensor([3 * alpha, (-3 ** 0.5) * alpha], device=x.device)[None, :]) ** 2, dim=-1) / (2 * 1 ** 2)) +
                torch.exp(torch.sum(-(x - torch.tensor([-3 * alpha, (-3 ** 0.5) * alpha], device=x.device)[None, :]) ** 2, dim=-1) / (2 * 1 ** 2)) +
                torch.exp(torch.sum(-(x - torch.tensor([0 * alpha, (2 * 3 ** 0.5) * alpha], device=x.device)[None, :]) ** 2, dim=-1) / (2 * 1 ** 2))) / 3

    with torch.no_grad():
        interval = 10
        timestep = T - 1 - 110

        branching_param_list = [(1, 1), (2, 2), (3, 2), (4, 2), (2, 4)]   # branching occurance, branching factor
        branching_nfe_list = [1, (2+4)/2/4, (2+4+8)/3/8, (2+4+8+16)/4/16, (4+16)/2/16]
        branching_nfe_list = [np.log(f)/np.log(2) for f in branching_nfe_list]
        mean_hybrid_rmse_list_combined = [[] for _ in branching_param_list]
        mean_tempflow_rmse_list = []

        next_idx = timestep
        s = torch.sqrt((1 - process.alpha_prev_bars[next_idx]) / (1 - process.alpha_bars[timestep]) * (1 - process.alpha_bars[timestep] / process.alpha_prev_bars[next_idx]))
        c = torch.sqrt((1 - process.alpha_bars[timestep]) * process.alpha_prev_bars[next_idx] / process.alpha_bars[timestep]) - torch.sqrt(1 - process.alpha_prev_bars[next_idx] - s ** 2)
        omega = s ** 2 / (c * torch.sqrt(1 - process.alpha_bars[timestep]))


        def get_value_grad(fp, n_samples, normalize=False, resume_t=None, branch_point=None, ode=False, use_r_tilde=False):
            xT = fp.to(device=device).squeeze()
            all_xt, all_mean, all_sigma, all_t, all_alpha_bar, all_alpha_bar_pred, all_eps = process.sampling(
                n_samples, only_final=False, interval=interval, xT=xT, resume_t=resume_t, branch_point=branch_point, ode=ode)  # T*batch*2 or batch*2
            all_reward = reward_function(all_xt[-1])  # batch

            if use_r_tilde:
                predict_epsilon_ref = model_ref(all_xt[:-1], all_t[:, None])
                x0_hat_ref = torch.sqrt(1 / all_alpha_bar[:, None, None]) * (all_xt[:-1] - torch.sqrt(1 - all_alpha_bar[:, None, None]) * predict_epsilon_ref)
                all_mean_ref = torch.sqrt(all_alpha_bar_pred[:, None, None]) * x0_hat_ref + torch.sqrt(1 - all_alpha_bar_pred[:, None, None] - all_sigma[:, None, None] ** 2) * predict_epsilon_ref
                loss_kl = ((all_mean - all_mean_ref) ** 2 / (2 * all_sigma.clamp(min=1e-3)[:, None, None] ** 2)).mean(dim=-1).sum(dim=0)
                all_reward = all_reward - loss_kl
            if normalize:   all_reward = (all_reward - all_reward.mean())
            estimations = all_reward[:, None] * all_eps[0] / all_sigma[0]
            return estimations.mean(dim=0)*omega


        datapoints_numpy = np.load('./tilted_dataset.npy')
        gaussian_numpy = np.load('./gaussian.npy')


        n_fixed_points = 100
        mbs_list = [4, 6, 8, 10, 12, 14, 16]


        for idx in tqdm(range(n_fixed_points)):
            data = torch.tensor(datapoints_numpy[idx]).to(device, dtype=torch.float32)
            gaussian = torch.tensor(gaussian_numpy[idx]).to(device, dtype=torch.float32)
            fixed_point = torch.sqrt(model.alpha_bars[timestep]) * data + torch.sqrt(1 - model.alpha_bars[timestep]) * gaussian

            with torch.enable_grad():
                fixed_point_g = fixed_point.unsqueeze(0).requires_grad_(True)
                opt_score = torch.autograd.grad(outputs=torch.log(opt_density(fixed_point_g, timestep)), inputs=fixed_point_g)[0]
                ref_score = torch.autograd.grad(outputs=torch.log(ref_density(fixed_point_g, timestep)), inputs=fixed_point_g)[0]
                gt_value_grad = opt_score - ref_score


            # branchgrpo
            for i, (n_split, n_branch) in enumerate(branching_param_list):
                n_endpoints = n_branch ** n_split
                branch_points = torch.tensor([timestep - timestep // n_split * i for i in range(n_split)])
                branch_point_info = (branch_points, n_endpoints, n_branch) if n_endpoints != 1 else None

                hybrid_rmse_list = []
                for mbs in mbs_list:
                    xT = fixed_point.to(device=device).squeeze()
                    all_xt, all_mean_ref, all_sigma, all_t, all_alpha_bar, all_alpha_bar_pred, all_eps = process.sampling(2 ** mbs // n_endpoints, only_final=False, interval=interval, xT=xT, resume_t=timestep, terminal_t=None, branch_point=branch_point_info)  # T*batch*2 or batch*2
                    all_reward = reward_function(all_xt[-1])
                    all_reward = (all_reward - all_reward.mean().detach())  # / (all_reward.std() + 1e-8)
                    est_value_grad = all_reward[:, None] * all_eps[0] / all_sigma[0] * omega

                    est_value_grad = est_value_grad.mean(dim=0)

                    hybrid_rmse_list.append(((est_value_grad - gt_value_grad) ** 2).mean().item() ** 0.5)
                mean_hybrid_rmse_list_combined[i].append(hybrid_rmse_list)


            # tempflowgrpo
            gt_value_grad_tempflow = get_value_grad(fixed_point, n_samples=minibatch_size, normalize=True, resume_t=timestep, ode=True)

            tempflow_rmse_list = []
            for mbs in mbs_list:
                xT = fixed_point.to(device=device).squeeze()
                all_xt, all_mean_ref, all_sigma, all_t, all_alpha_bar, all_alpha_bar_pred, all_eps = process.sampling(
                    2 ** mbs, only_final=False, interval=interval, xT=xT, resume_t=timestep, terminal_t=None, branch_point=None, ode=True)  # T*batch*2 or batch*2
                all_reward = reward_function(all_xt[-1])
                all_reward = (all_reward - all_reward.mean().detach())  # / (all_reward.std() + 1e-8)
                est_value_grad = all_reward[:, None] * all_eps[0] / all_sigma[0]
                est_value_grad = est_value_grad.mean(dim=0) * omega
                tempflow_rmse_list.append(((est_value_grad - gt_value_grad_tempflow) ** 2).mean().item() ** 0.5)
            mean_tempflow_rmse_list.append(tempflow_rmse_list)


        mean_hybrid_rmse_list_combined = [np.mean(np.array(mean_hybrid_rmse_list), axis=0) for mean_hybrid_rmse_list in mean_hybrid_rmse_list_combined]
        mean_tempflow_rmse_list = np.mean(np.array(mean_tempflow_rmse_list), axis=0)
        mean_hybrid_rmse_list_combined = np.array(mean_hybrid_rmse_list_combined)



        # plotting
        seaborn.set_theme(style="white", rc={"xtick.bottom": True, "ytick.left": True, })
        plt.figure(figsize=(4, 3))

        mbs_list = np.array(mbs_list)
        for ratio, mean_hybrid_rmse_list, factor in reversed(list(zip(branching_param_list, mean_hybrid_rmse_list_combined, branching_nfe_list))):
            if mode == 'tempflowgrpo':
                if ratio == (1, 1):
                    plt.plot(mbs_list+factor, mean_hybrid_rmse_list, marker='X', label=f'SDE', linewidth=2.5, markersize=6, zorder=4, linestyle='-', alpha=0.8)#, color=c)
            else:
                plt.plot(mbs_list+factor, mean_hybrid_rmse_list, marker='X', label=f'#split:{ratio[0]}, #branch:{ratio[1]}', linewidth=2, markersize=6, zorder=4, linestyle='--', alpha=0.8)#, color=c)
        if mode == 'tempflowgrpo':
            plt.plot(mbs_list, mean_tempflow_rmse_list, marker='X', label='SDE&ODE mixed', linewidth=2.5, markersize=6, zorder=4, linestyle='-', alpha=0.8)
        plt.axhline(y=0.0, color='#556', linestyle='-', zorder=1, linewidth=2.5)
        ax = plt.gca()
        ax.text(-0.05, 0.35, r'RMSE ($\bar{\alpha}_t$=0.01)', transform=ax.get_yaxis_transform(), rotation=90, va='center', ha='center', fontsize=14)
        plt.xticks(mbs_list, fontsize=12)
        plt.yticks([0, 0.7], ['0', '0.7'], fontsize=14)
        plt.ylim(-0.03, 0.72)
        if mode == 'tempflowgrpo':
            ax.text(10, -0.08, r'log2(#samples)', transform=ax.get_xaxis_transform(), rotation=0, va='center', ha='center', fontsize=14)
        else:
            ax.text(10, -0.08, r'log2(NFE/40)', transform=ax.get_xaxis_transform(), rotation=0, va='center', ha='center', fontsize=14)
        plt.xticks([4, 16], ['4', '16'], fontsize=14)
        plt.legend(frameon=False, fontsize=12, loc='upper right')
        plt.tight_layout()
        if mode == 'tempflowgrpo':
            plt.savefig('branch_zeroth_tempflowgrpo.png', dpi=300)
        else:
            plt.savefig('branch_zeroth_branchgrpo.png', dpi=300)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="branchgrpo", choices=['branchgrpo', 'tempflowgrpo'])
    args = parser.parse_args()

    main(args)