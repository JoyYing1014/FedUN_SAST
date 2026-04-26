import sast as fs
import torch
from sast.lightsecagg.SecAggMath import SecAggMath

class FedUN_SecAgg_Client(fs.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.secagg_math = SecAggMath(bit_length=31) # 强制对齐 31
        self.stored_Z_shares = {}
        self.stored_delta_shares = {}

    def get_message(self, msg):
        return_msg = {}

        if msg['command'] == 'cal_secagg_update':
            U_t = msg.get('U_t', None)
            C_clip = msg.get('C_clip', 1.0)
            beta = msg.get('beta', 1.0)
            gamma = msg.get('gamma', 5.0)
            do_projection = msg.get('do_projection', True)
            target_module = msg['target_module']

            if self.step_type == 'sgd':
                self.cal_gradient_loss_sgd(msg['epochs'], msg['lr'], target_module)
            else:
                self.cal_gradient_loss(msg['epochs'], msg['lr'], target_module)

            g_local = self.upload_grad
            n_i = self.local_training_number
            total_samples = msg.get('total_samples', n_i)
            p_i = float(n_i) / float(total_samples)

            Z_i, delta_i = None, None
            if not self.unlearn_flag:
                norm_g = torch.norm(g_local)
                clip_factor = min(1.0, float(C_clip) / (float(norm_g) + 1e-6))
                g_bar = g_local * clip_factor
                if U_t is not None:
                    c_i = g_bar @ U_t
                    Z_i = torch.outer(g_bar, c_i)
                delta_i = p_i * beta * g_local
            else:
                if U_t is not None and do_projection:
                    coeff = U_t.T @ g_local
                    proj = U_t @ coeff
                    r = g_local - proj
                    d_u = (torch.norm(g_local) / torch.norm(r)) * r if torch.norm(r) > 1e-6 else torch.zeros_like(g_local)
                else:
                    d_u = g_local
                delta_i = p_i * gamma * d_u

            use_secagg = msg.get('use_secagg', True)
            if use_secagg:
                U_lsa, T_lsa, N_lsa = msg['U_lsa'], msg['T_lsa'], msg['N_lsa']
                W_lsa = msg['W_lsa'].to(self.device)

                Z_i_cipher, Z_shares, Z_meta = None, None, None
                if Z_i is not None:
                    Z_i_fq = self.secagg_math.quantize_to_finite_field(Z_i)
                    self.mask_Z_fq = self.secagg_math.generate_mask_in_fq(Z_i_fq.shape, self.device)
                    Z_i_cipher = torch.remainder(Z_i_fq + self.mask_Z_fq, self.secagg_math.q)
                    Z_shares, shape, pad = self.secagg_math.lightsecagg_encode(self.mask_Z_fq, U_lsa, T_lsa, N_lsa, W_lsa, self.device)
                    Z_meta = (shape, pad)

                delta_i_fq = self.secagg_math.quantize_to_finite_field(delta_i)
                self.mask_Delta_fq = self.secagg_math.generate_mask_in_fq(delta_i_fq.shape, self.device)
                delta_i_cipher = torch.remainder(delta_i_fq + self.mask_Delta_fq, self.secagg_math.q)
                delta_shares, d_shape, d_pad = self.secagg_math.lightsecagg_encode(self.mask_Delta_fq, U_lsa, T_lsa, N_lsa, W_lsa, self.device)
                delta_meta = (d_shape, d_pad)

                return_msg.update({
                    'Z_i_cipher': Z_i_cipher, 'delta_i_cipher': delta_i_cipher,
                    'Z_shares': Z_shares, 'delta_shares': delta_shares,
                    'Z_meta': Z_meta, 'delta_meta': delta_meta, 'n_i': n_i
                })
            else:
                return_msg.update({'Z_i_cipher': Z_i, 'delta_i_cipher': delta_i, 'n_i': n_i})
            return return_msg

        elif msg['command'] == 'store_shares':
            self.stored_Z_shares = msg.get('Z_shares', {})
            self.stored_delta_shares = msg.get('delta_shares', {})
            return {'status': 'success'}

        elif msg['command'] == 'aggregate_shares':
            surviving_indices = msg['surviving_indices']
            Z_agg, delta_agg = None, None

            if self.stored_Z_shares and any(i in self.stored_Z_shares for i in surviving_indices):
                first_share = next(iter(self.stored_Z_shares.values()))
                Z_agg = torch.zeros_like(first_share)
                for i in surviving_indices:
                    if i in self.stored_Z_shares:
                        Z_agg = torch.remainder(Z_agg + self.stored_Z_shares[i], self.secagg_math.q)

            if self.stored_delta_shares:
                first_share = next(iter(self.stored_delta_shares.values()))
                delta_agg = torch.zeros_like(first_share)
                for i in surviving_indices:
                    if i in self.stored_delta_shares:
                        delta_agg = torch.remainder(delta_agg + self.stored_delta_shares[i], self.secagg_math.q)

            return {'Z_agg': Z_agg, 'delta_agg': delta_agg}

        return super().get_message(msg)
