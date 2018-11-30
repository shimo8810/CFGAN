import chainer
import chainer.functions as F
class CFGANZRUpdater(chainer.training.updaters.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(CFGANZRUpdater, self).__init__(*args, **kwargs)

    def loss_gen(self, gen, y_fake, r_u, r_u_fake, negative_idx, alpha=0.1):
        batchsize = len(y_fake)
        loss_adv = F.sum(F.softplus(-y_fake)) / batchsize
        loss_zr = F.mean_squared_error(r_u_fake[negative_idx], r_u[negative_idx])
        loss = loss_adv + alpha * loss_zr
        chainer.report({'loss': loss}, gen)
        return loss

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        loss_real = F.sum(F.softplus(-y_real)) / batchsize
        loss_fake = F.sum(F.softplus( y_fake)) / batchsize
        loss = loss_fake + loss_real
        chainer.report({'loss': loss}, dis)
        return loss

    def get_negative_items_list(self, data, xp, S=0.5):
        user_miss, item_miss = xp.where(data == 0)
        n_user = len(data)
        negative_items_list = []
        negative_users_list = []
        for user in range(n_user):
            missings = item_miss[user_miss == 0]
            negative_items = xp.random.choice(missings,
                            size=int(S*len(missings)), replace=False)
            negative_items_list.append(negative_items)
            negative_users_list.append(xp.ones_likes(negative_items) * user)
        user_idx = xp.hstack(negative_users_list)
        item_idx = xp.hstack(negative_items_list)
        return user_idx, item_idx

    def update_core(self):
        gen_opt = self.get_optimizer('gen')
        dis_opt = self.get_optimizer('dis')

        gen, dis = self.gen, self.dis
        xp = gen.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        r_u, c_u = chainer.dataset.concat_examples(batch, self.device)
        e_u = xp.where(r_u > 0, 1, 0)
        z = xp.array(gen.make_z(batchsize))
        negative_idx = self.get_negative_items_list(r_u, xp)

        r_u_fake = gen(z, c_u)
        y_fake = dis(r_u_fake * e_u, c_u)
        y_real = dis(r_u, c_u)

        dis_opt.update(self.loss_dis, dis, y_fake, y_real)
        gen_opt.update(self.loss_gen, gen, y_fake, r_u, r_u_fake, negative_idx)