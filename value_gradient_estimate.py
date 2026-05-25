import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

import seaborn
import einops



class Backbone(nn.Module):
    def __init__(self, n_steps, input_dim = 2):
        super().__init__()
        self.linear_model1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Dropout(0.2),
            nn.GELU()
        )
        # Condition time t
        self.embedding_layer = nn.Embedding(n_steps, 256)

        self.linear_model2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Linear(512, 512),
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Linear(512, input_dim),
        )
    def forward(self, x, idx):
        x = self.linear_model2(self.linear_model1(x) + self.embedding_layer(idx))
        return x

class Model(nn.Module):
    def __init__(self, device, beta_1, beta_T, T, input_dim):
        '''
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        input_dim : a dimension of data
        '''
        super().__init__()
        self.device = device
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start = beta_1, end=beta_T, steps=T), dim = 0).to(device = device)
        self.backbone = Backbone(T, input_dim)

        self.to(device = self.device)

    def loss_fn(self, x, idx=None):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index. Else (inference phase), it is recommended that you specify.
        '''
        output, epsilon, alpha_bar = self.forward(x, t=idx, get_target=True)
        loss = (output - epsilon).square().mean()
        return loss


    def forward(self, x, t=None, get_target=False):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index. Else (inference phase), it is recommended that you specify.
        get_target : if True (training phase), target and sigma is returned with output (epsilon prediction)
        '''

        if t == None:
            t = torch.randint(0, len(self.alpha_bars), (x.size(0), )).to(device = self.device)
            used_alpha_bars = self.alpha_bars[t][:, None]
            epsilon = torch.randn_like(x)
            x_tilde = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon

        else:
            if type(t) == int:  t = torch.Tensor([t for _ in range(x.size(0))]).to(device = self.device).long()
            x_tilde = x

        output = self.backbone(x_tilde, t)
        return (output, epsilon, used_alpha_bars) if get_target else output

class DiffusionProcess():
    def __init__(self, beta_1, beta_T, T, diffusion_fn, device, shape):
        '''
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        diffusion_fn  : trained diffusion network
        shape         : data shape
        '''

        self.betas = torch.linspace(start = beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start = beta_1, end=beta_T, steps=T), dim = 0).to(device = device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        self.shape = shape

        self.diffusion_fn = diffusion_fn
        self.device = device

    def _one_diffusion_step(self, xt, interval=1, resume_t=None, terminal_t=None, branch_point=None, ode=False):
        all_idx = list(reversed(range(len(self.alpha_bars))))[::interval]
        is_first_step = True
        for idx in all_idx:
            if idx > resume_t:  continue
            if idx <= terminal_t: break
            predict_epsilon = self.diffusion_fn(xt, idx)
            next_idx = idx-interval+1

            if ode and not is_first_step:   sigma = torch.tensor(0.0).to(device = self.device)
            else:   sigma = torch.sqrt((1 - self.alpha_prev_bars[next_idx]) / (1 - self.alpha_bars[idx]) * (1-self.alpha_bars[idx] / self.alpha_prev_bars[next_idx]))
            is_first_step = False
            # print(self.alpha_bars[idx], self.alpha_prev_bars[next_idx], sigma)

            x0_hat = torch.sqrt(1 / self.alpha_bars[idx]) * (xt - torch.sqrt(1-self.alpha_bars[idx]) * predict_epsilon)
            xt_mean = torch.sqrt(self.alpha_prev_bars[next_idx]) * x0_hat + torch.sqrt(1 - self.alpha_prev_bars[next_idx] - sigma**2) * predict_epsilon
            #xt_mean = torch.sqrt(self.alpha_prev_bars[next_idx]/self.alpha_bars[idx]) * (xt - (1-self.alpha_bars[idx]/self.alpha_prev_bars[next_idx])/(1-self.alpha_bars[idx])**0.5 * predict_epsilon)

            if branch_point is not None:
                unbranched_count = torch.sum(idx > branch_point[0]).item()
                #print(unbranched_count, branch_point[1], branch_point[2])
                noise = torch.randn([xt.shape[0]//(branch_point[2]**unbranched_count), *xt.shape[1:]]).to(device = self.device)
                #print(f'branching at idx {idx}, duplicate: {2**duplicate}, noise shape: {noise.shape}', branch_point)
                noise = einops.repeat(noise, 'b ... -> (r b) ...', r=branch_point[2]**unbranched_count)
            else:
                noise = torch.randn_like(xt)
            xt = xt_mean + sigma * noise

            yield xt, (idx, xt_mean, sigma, self.alpha_bars[idx], self.alpha_prev_bars[next_idx], noise)

    def sampling(self, sampling_number, only_final=False, interval=1, xT=None, resume_t=None, terminal_t=None, branch_point=None, ode=False):
        '''
        sampling_number : a number of generation
        only_final      : If True, return is an only output of final schedule step
        '''
        if resume_t is None:    resume_t = len(self.alpha_bars)-1
        if terminal_t is None:  terminal_t = -1
        if xT is not None:
            if xT.ndim == 1:    xt = xT.repeat(sampling_number, *[1 for _ in self.shape]).to(device = self.device)#.squeeze()
            else:   xt = xT.to(device = self.device).squeeze()
        else:
            xt = torch.randn([sampling_number,*self.shape]).to(device = self.device).squeeze()
        if branch_point is not None:
            duplicate_number = branch_point[1]
            xt = einops.repeat(xt, 'b ... -> (r b) ...', r=duplicate_number)
        xt_list = [xt]
        xtm1_mean_list = []
        sigma_list = []
        t_list = []
        alpha_bar_list = []
        alpha_bar_prev_list = []
        noise_list = []

        for idx, (xt, aux) in enumerate(self._one_diffusion_step(xt, interval=interval, resume_t=resume_t, terminal_t=terminal_t, branch_point=branch_point, ode=ode)):
            xt_list.append(xt)

            t, xt_mean, sigma, alpha_bar, alpha_bar_prev, noise = aux
            xtm1_mean_list.append(xt_mean)
            sigma_list.append(sigma)
            t_list.append(t)
            alpha_bar_list.append(alpha_bar)
            alpha_bar_prev_list.append(alpha_bar_prev)
            noise_list.append(noise)

        if only_final:  return xt
        return (torch.stack(xt_list), torch.stack(xtm1_mean_list), torch.stack(sigma_list), torch.tensor(t_list).to(device = self.device).long(),
                torch.stack(alpha_bar_list), torch.stack(alpha_bar_prev_list), torch.stack(noise_list))

class GaussianDataset(torch.utils.data.Dataset):
    def __init__(self, dist1, dist2, dist3, shape = (2), probability=(0.33, 0.66), total_len = 1000000):
        self.dist1_mean, self.dist1_var = torch.tensor([dist1[0], dist1[1]]), dist1[2]
        self.dist2_mean, self.dist2_var = torch.tensor([dist2[0], dist2[1]]), dist2[2]
        self.dist3_mean, self.dist3_var = torch.tensor([dist3[0], dist3[1]]), dist3[2]
        self.shape = shape
        self.probability = probability
        self.total_len = total_len

    @property
    def get_probability(self):
        tmp = torch.rand(1)
        if tmp < self.probability[0]:   return 0
        elif tmp < self.probability[1]: return 1
        else:   return 2
        # return torch.rand(1) < self.probability

    @property
    def _sampling_1(self):
        return self.dist1_mean + torch.randn(self.shape) * self.dist1_var

    @property
    def _sampling_2(self):
        return self.dist2_mean + torch.randn(self.shape) * self.dist2_var

    @property
    def _sampling_3(self):
        return self.dist3_mean + torch.randn(self.shape) * self.dist3_var

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        cls = self.get_probability
        if cls == 0:
            data = self._sampling_1
        elif cls == 1:
            data = self._sampling_2
        else:
            data = self._sampling_3
        #data = self._sampling_1 if cls == 0 else self._sampling_2
        return data



def reward_function(x):
    return -x[:, 0]/2+3



def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    beta_1 = 1e-4
    beta_T = 0.03
    T = 500
    shape = (2,)

    device = torch.device('cuda')
    model = Model(device, beta_1, beta_T, T, shape[0])
    model.load_state_dict(torch.load('./model_ft.pth'))
    model.eval()
    process = DiffusionProcess(beta_1, beta_T, T, model, device, shape=shape)

    dist1, dist2, dist3 = (3, -3 ** 0.5, 1), (-3, -3 ** 0.5, 1), (0, 2 * 3 ** 0.5, 1)
    dataset = GaussianDataset(dist1, dist2, dist3, probability=(0.33, 0.66), total_len=1000000)
    dist1, dist2, dist3 = (3 - 0.5, -3 ** 0.5, 1), (-3 - 0.5, -3 ** 0.5, 1), (0 - 0.5, 2 * 3 ** 0.5, 1)
    tilted_dataset = GaussianDataset(dist1, dist2, dist3, probability=(1 / (math.exp(3) + math.exp(1.5) + 1), 1 - math.exp(1.5) / (math.exp(3) + math.exp(1.5) + 1)), total_len=10000)

    model_ref = Model(device, beta_1, beta_T, T, shape[0])
    model_ref.load_state_dict(torch.load('./model_base.pth'))
    model_ref.eval()

    datapoints_numpy = np.array([tilted_dataset[i] for i in range(10000)])
    np.save('./tilted_dataset.npy', datapoints_numpy)
    gaussian_numpy = np.random.randn(10000, 2)
    np.save('./gaussian.npy', gaussian_numpy)

    def opt_density(x, t):
        alpha = process.alpha_bars[t] ** 0.5
        return (torch.exp(torch.sum(-(x - torch.tensor([(3 - 0.5) * alpha, (-3 ** 0.5 -.0) * alpha], device=x.device)[None, :]) ** 2, dim=-1) / (2 * 1 ** 2)) * math.exp(-1.5)+
                torch.exp(torch.sum(-(x - torch.tensor([(-3 - 0.5) * alpha, (-3 ** 0.5 -.0) * alpha], device=x.device)[None, :]) ** 2, dim=-1) / (2 * 1 ** 2)) * math.exp(1.5)+
                torch.exp(torch.sum(-(x - torch.tensor([(0 - 0.5) * alpha, (2 * 3 ** 0.5 - .0) * alpha], device=x.device)[None, :]) ** 2, dim=-1) / (2 * 1 ** 2)) ) / (1+math.exp(-1.5)+math.exp(1.5))
    def ref_density(x, t):
        alpha = process.alpha_bars[t] ** 0.5
        return (torch.exp(torch.sum(-(x - torch.tensor([3 * alpha, (-3 ** 0.5) * alpha], device=x.device)[None, :]) ** 2, dim=-1) / (2 * 1 ** 2)) +
                torch.exp(torch.sum(-(x - torch.tensor([-3 * alpha, (-3 ** 0.5) * alpha], device=x.device)[None, :]) ** 2, dim=-1) / (2 * 1 ** 2)) +
                torch.exp(torch.sum(-(x - torch.tensor([0 * alpha, (2 * 3 ** 0.5) * alpha], device=x.device)[None, :]) ** 2, dim=-1) / (2 * 1 ** 2)) ) / 3

    with torch.no_grad():
        interval = 10
        timestep = T-1-110

        mean_ppo_rmse_list = []
        mean_grpo_rmse_list = []
        mean_dps_rmse_list = []
        mean_sqdf2_rmse_list = []

        next_idx = timestep
        s = torch.sqrt((1 - process.alpha_prev_bars[next_idx]) / (1 - process.alpha_bars[timestep]) * (1 - process.alpha_bars[timestep] / process.alpha_prev_bars[next_idx]))
        c = torch.sqrt((1 - process.alpha_bars[timestep]) * process.alpha_prev_bars[next_idx] / process.alpha_bars[timestep]) - torch.sqrt(1 - process.alpha_prev_bars[next_idx] - s ** 2)
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

        for idx in range(50):
            data = torch.tensor(datapoints_numpy[idx]).to(device, dtype=torch.float32)
            gaussian = torch.tensor(gaussian_numpy[idx]).to(device, dtype=torch.float32)
            fixed_point = torch.sqrt(model.alpha_bars[timestep]) * data + torch.sqrt(1 - model.alpha_bars[timestep]) * gaussian

            print("="*50)
            print('fixed point:\t\t', data, gaussian, fixed_point.squeeze())

            with torch.enable_grad():
                fixed_point_g = fixed_point.unsqueeze(0).requires_grad_(True)
                opt_score = torch.autograd.grad(outputs=torch.log(opt_density(fixed_point_g, timestep)), inputs=fixed_point_g)[0]
                ref_score = torch.autograd.grad(outputs=torch.log(ref_density(fixed_point_g, timestep)), inputs=fixed_point_g)[0]
                gt_value_grad = opt_score-ref_score
            print('ground truth:\t\t', gt_value_grad.squeeze())

            # PPO
            ppo_rmse_list = []
            for mbs in mbs_list:
                est_value_grad = get_value_grad(fixed_point, n_samples=2**mbs, resume_t=timestep)
                ppo_rmse_list.append(((est_value_grad - gt_value_grad)**2).mean().item()**0.5)
            mean_ppo_rmse_list.append(ppo_rmse_list)

            # GRPO
            grpo_rmse_list = []
            for mbs in mbs_list:
                est_value_grad = get_value_grad(fixed_point, n_samples=2**mbs, normalize=True, resume_t=timestep)
                grpo_rmse_list.append(((est_value_grad - gt_value_grad)**2).mean().item()**0.5)
            print('grpo_value_grad:\t', est_value_grad)

            # DPS
            with torch.enable_grad():
                xT = fixed_point.detach().clone().to(device = device).unsqueeze(0)
                xT.requires_grad_(True)
                predict_epsilon_base = model(xT, timestep)
                x0_hat = torch.sqrt(1 / model.alpha_bars[timestep]) * (xT - torch.sqrt(1 - model.alpha_bars[timestep]) * predict_epsilon_base)
                dps_value_grad = torch.autograd.grad(outputs=reward_function(x0_hat), inputs=xT)[0]

            # SQDF-2
            with torch.enable_grad():
                xT = fixed_point.detach().clone().to(device = device).unsqueeze(0)
                xT.requires_grad_(True)
                n_step = 2
                timesteps = [timestep - timestep // n_step * i for i in range(n_step)]
                predict_epsilon = model(xT, timesteps[0])
                x0_hat = torch.sqrt(1 / model.alpha_bars[timesteps[0]]) * (xT - torch.sqrt(1 - model.alpha_bars[timesteps[0]]) * predict_epsilon)
                xT2 = torch.sqrt(model.alpha_bars[timesteps[1]]) * x0_hat + torch.sqrt(1 - model.alpha_bars[timesteps[1]]) * predict_epsilon
                predict_epsilon2 = model(xT2, timesteps[1])
                x0_hat2 = torch.sqrt(1 / model.alpha_bars[timesteps[1]]) * (xT2 - torch.sqrt(1 - model.alpha_bars[timesteps[1]]) * predict_epsilon2)
                sqdf2_value_grad = torch.autograd.grad(outputs=reward_function(x0_hat2), inputs=xT)[0]

            mean_grpo_rmse_list.append(grpo_rmse_list)
            mean_dps_rmse_list.append( ((dps_value_grad.squeeze() - gt_value_grad)**2).mean().item()**0.5 )
            mean_sqdf2_rmse_list.append( ((sqdf2_value_grad.squeeze() - gt_value_grad)**2).mean().item()**0.5 )

        mean_ppo_rmse_list = np.mean(np.array(mean_ppo_rmse_list), axis=0)
        mean_grpo_rmse_list = np.mean(np.array(mean_grpo_rmse_list), axis=0)
        mean_dps_rmse = np.mean(np.array(mean_dps_rmse_list), axis=0)
        mean_sqdf2_rmse = np.mean(np.array(mean_sqdf2_rmse_list), axis=0)

        seaborn.set_theme(style="white", rc={"xtick.bottom": True, "ytick.left": True, })
        plt.figure(figsize=(4, 3))
        plt.axhline(y=0.0, color='#555', linestyle='-', zorder=1)
        plt.plot(mbs_list, mean_ppo_rmse_list, marker='X', label='ZO: w/o control variate', color='#40c090', linewidth=2.5, markersize=6, zorder=3)
        plt.plot(mbs_list, mean_grpo_rmse_list, marker='X', label='ZO: w/ control variate', color='#6685cc', linewidth=2.5, markersize=6, zorder=3)

        plt.axhline(y=mean_dps_rmse, color='#bbc', linestyle=(0, (7, 1)), label=r'FO: $\nabla_{x_t}r(x_{0|t})$', linewidth=2, zorder=2.1)
        plt.axhline(y=mean_sqdf2_rmse, color='#aab', linestyle=(0, (2.5, 1)), label=r'FO: SQDF$_{2stepODE}$', linewidth=3.5, zorder=2) #f09005, e04949, cb3
        ax = plt.gca()
        ax.text(-0.05, 2.0, r'RMSE($\bar{\alpha}_t$=0.01)', transform=ax.get_yaxis_transform(), rotation=90, va='center', ha='center', fontsize=14)
        plt.yticks([0, 4.0], ['0', '4.0'], fontsize=14)
        ax.text(10, -0.08, r'log2(#samples)', transform=ax.get_xaxis_transform(), rotation=0, va='center', ha='center', fontsize=14)
        plt.xticks([4, 16], ['4', '16'], fontsize=14)
        plt.ylim(-0.1, 4.1)
        plt.legend(frameon=False, fontsize=12, loc='upper left')
        plt.tight_layout()
        plt.savefig('value_gradient_estimate_plot.png', dpi=300)



if __name__ == '__main__':
    main()