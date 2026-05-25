import torch
import math
import matplotlib.pyplot as plt
import numpy as np

import seaborn
from tqdm import tqdm
from value_gradient_estimate import Model, GaussianDataset, reward_function, DiffusionProcess



def main(args):
    mode = args.mode  # 'first' or 'zeroth'

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    beta_1 = 1e-4
    beta_T = 0.03
    T = 500
    shape = (2,)


    device = torch.device('cuda')
    model = Model(device, beta_1, beta_T, T, shape[0])
    process = DiffusionProcess(beta_1, beta_T, T, model, device, shape=shape)

    model.load_state_dict(torch.load('./model_ft.pth'))
    model.eval()

    model_ref = Model(device, beta_1, beta_T, T, shape[0])
    model_ref.load_state_dict(torch.load('./model_base.pth'))
    model_ref.eval()

    dist1, dist2, dist3 = (3 - 0.5, -3 ** 0.5, 1), (-3 - 0.5, -3 ** 0.5, 1), (0 - 0.5, 2 * 3 ** 0.5, 1)
    tilted_dataset = GaussianDataset(dist1, dist2, dist3, probability=(1 / (math.exp(3) + math.exp(1.5) + 1), 1 - math.exp(1.5) / (math.exp(3) + math.exp(1.5) + 1)), total_len=100)

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

        mean_grpo_rmse_list = []
        mean_hybrid_ratio_list = [0.9, 0.8, 0.5]
        mean_hybrid_rmse_list_combined = [[] for _ in mean_hybrid_ratio_list]
        mean_dps_rmse_list = []

        next_idx = timestep
        s = torch.sqrt((1 - process.alpha_prev_bars[next_idx]) / (1 - process.alpha_bars[timestep]) * (1 - process.alpha_bars[timestep] / process.alpha_prev_bars[next_idx]))
        c = torch.sqrt((1 - process.alpha_bars[timestep]) * process.alpha_prev_bars[next_idx] / process.alpha_bars[timestep]) - torch.sqrt(
            1 - process.alpha_prev_bars[next_idx] - s ** 2)
        omega = s ** 2 / (c * torch.sqrt(1 - process.alpha_bars[timestep]))

        def get_value_grad(fp, n_samples, normalize=False, resume_t=None, branch_point=None, use_r_tilde=False):
            xT = fp.to(device = device).squeeze()
            all_xt, all_mean, all_sigma, all_t, all_alpha_bar, all_alpha_bar_pred, all_eps = process.sampling(
                n_samples, only_final=False, interval=interval, xT=xT, resume_t=resume_t, branch_point=branch_point)  # T*batch*2 or batch*2
            all_reward = reward_function(all_xt[-1])  # batch

            if use_r_tilde:
                predict_epsilon_ref = model_ref(all_xt[:-1], all_t[:, None])
                x0_hat_ref = torch.sqrt(1 / all_alpha_bar[:, None, None]) * (all_xt[:-1] - torch.sqrt(1 - all_alpha_bar[:, None, None]) * predict_epsilon_ref)
                all_mean_ref = torch.sqrt(all_alpha_bar_pred[:, None, None]) * x0_hat_ref + torch.sqrt(1 - all_alpha_bar_pred[:, None, None] - all_sigma[:, None, None] ** 2) * predict_epsilon_ref
                loss_kl = ((all_mean - all_mean_ref) ** 2 / (2 * all_sigma.clamp(min=1e-3)[:,None, None] ** 2)).mean(dim=-1).sum(dim=0)
                all_reward = all_reward - loss_kl
            if normalize:   all_reward = (all_reward - all_reward.mean())

            estimations = all_reward[:, None] * all_eps[0] / all_sigma[0]
            return estimations.mean(dim=0)*omega


        datapoints_numpy = np.load('./tilted_dataset.npy')
        gaussian_numpy = np.load('./gaussian.npy')
        mbs_list = [4, 6, 8, 10, 12, 14, 16]

        for idx in tqdm(range(100)):
            data = torch.tensor(datapoints_numpy[idx]).to(device, dtype=torch.float32)
            gaussian = torch.tensor(gaussian_numpy[idx]).to(device, dtype=torch.float32)

            fixed_point = torch.sqrt(model.alpha_bars[timestep]) * data + torch.sqrt(1 - model.alpha_bars[timestep]) * gaussian


            with torch.enable_grad():
                fixed_point_g = fixed_point.unsqueeze(0).requires_grad_(True)
                opt_score = torch.autograd.grad(outputs=torch.log(opt_density(fixed_point_g, timestep)), inputs=fixed_point_g)[0]
                ref_score = torch.autograd.grad(outputs=torch.log(ref_density(fixed_point_g, timestep)), inputs=fixed_point_g)[0]
                gt_value_grad = opt_score - ref_score


            # GRPO
            grpo_rmse_list = []
            for mbs in mbs_list:
                est_value_grad = get_value_grad(fixed_point, n_samples=2 ** mbs, normalize=True, resume_t=timestep)
                grpo_rmse_list.append(((est_value_grad - gt_value_grad) ** 2).mean().item() ** 0.5)
            mean_grpo_rmse_list.append(grpo_rmse_list)

            # Hybrid
            for i, lookahead_stop_ratio in enumerate(mean_hybrid_ratio_list):
                hybrid_rmse_list = []
                for mbs in mbs_list:
                    condition = torch.enable_grad() if mode == 'first' else torch.no_grad()
                    with condition:
                        xT = fixed_point.to(device=device).squeeze()
                        xT.requires_grad_(True)
                        all_xt, all_mean_ref, all_sigma, all_t, all_alpha_bar, all_alpha_bar_pred, all_eps = process.sampling(
                            2 ** mbs, only_final=False, interval=interval, xT=xT, resume_t=timestep, terminal_t=int(timestep * lookahead_stop_ratio))  # T*batch*2 or batch*2
                        last_xt, last_t = all_xt[-1], all_t[-1] - interval  # batch*2, batch

                        predict_epsilon_base = model(last_xt, last_t)
                        x0_hat = torch.sqrt(1 / model.alpha_bars[last_t]) * (last_xt - torch.sqrt(1 - model.alpha_bars[last_t]) * predict_epsilon_base)
                        all_reward = reward_function(x0_hat)
                        all_reward = all_reward - all_reward.mean().detach()
                        # first order
                        if mode == 'first':  est_value_grad = torch.autograd.grad(outputs=all_reward.mean(), inputs=xT)[0]
                    # zeroth order
                    if mode == 'zeroth':    est_value_grad = (all_reward[:, None] * all_eps[0] / all_sigma[0]).mean(dim=0)

                    hybrid_rmse_list.append(((est_value_grad - gt_value_grad) ** 2).mean().item() ** 0.5)
                mean_hybrid_rmse_list_combined[i].append(hybrid_rmse_list)

            # DPS
            with torch.enable_grad():
                xT = fixed_point.detach().clone().to(device=device).unsqueeze(0)
                xT.requires_grad_(True)
                predict_epsilon_base = model(xT, timestep)
                x0_hat = torch.sqrt(1 / model.alpha_bars[timestep]) * (xT - torch.sqrt(1 - model.alpha_bars[timestep]) * predict_epsilon_base)
                dps_value_grad = torch.autograd.grad(outputs=reward_function(x0_hat), inputs=xT)[0]
            mean_dps_rmse_list.append(((dps_value_grad.squeeze() - gt_value_grad) ** 2).mean().item() ** 0.5)



        mean_grpo_rmse_list = np.mean(np.array(mean_grpo_rmse_list), axis=0)
        mean_hybrid_rmse_list_combined = [np.mean(np.array(mean_hybrid_rmse_list), axis=0) for mean_hybrid_rmse_list in mean_hybrid_rmse_list_combined]
        mean_dps_rmse = np.mean(np.array(mean_dps_rmse_list), axis=0)

        seaborn.set_theme(style="white", rc={"xtick.bottom": True, "ytick.left": True, })
        plt.figure(figsize=(4, 3))

        if mode == 'zeroth':    plt.plot(mbs_list, mean_grpo_rmse_list, marker='X', label=r'$t_j/t_i=$'+'0.00(w/ control variate)', color='#6685cc', linewidth=3, markersize=6, zorder=3) #5555cc
        color_list = ['#E04949', '#BA51D4', '#8553D0']
        for ratio, mean_hybrid_rmse_list, c in reversed(list(zip(mean_hybrid_ratio_list, mean_hybrid_rmse_list_combined, color_list))):
            plt.plot(mbs_list, mean_hybrid_rmse_list+0.0, marker='X', label=r'$t_j/t_i=$'+f'{ratio:.2f}', linewidth=2, markersize=6, zorder=4, color=c, linestyle='--', alpha=0.8)
        if mode == 'first':     plt.axhline(y=mean_dps_rmse, color='#bbc', linestyle=(0, (7, 1)), label=r'$t_j/t_i=$'+r'1.00($\nabla_{x_t}r(x_{0|t})$)', linewidth=3, zorder=2)

        plt.axhline(y=0.0, color='#556', linestyle='-', zorder=1, linewidth=2.5)
        ax = plt.gca()
        ax.text(-0.05, 0.35, r'RMSE ($\bar{\alpha}_t$=0.01)', transform=ax.get_yaxis_transform(), rotation=90, va='center', ha='center', fontsize=14)

        plt.yticks([0, 0.7], ['0', '0.7'], fontsize=14)
        plt.ylim(-0.03, 0.72)
        ax.text(10, -0.08, r'log2(#samples)', transform=ax.get_xaxis_transform(), rotation=0, va='center', ha='center', fontsize=14)
        plt.xticks([4, 16], ['4', '16'], fontsize=14)
        plt.legend(frameon=False, fontsize=12, loc='upper left')
        plt.tight_layout()
        plt.savefig(f'lookahead_{mode}.png', dpi=300)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="first")
    args = parser.parse_args()

    main(args)