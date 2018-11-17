import chainer
import chainer.functions as F
class CFGANZRUpdater(chainer.training.updaters.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(CFGANZRUpdater, self).__init__(*args, **kwargs)

    def loss_gen(self, gen, r_u_fake, e_u):
        batchsize = len(e_u)

    def loss_dis(self, dis, r_u_fake, r_u_real, e_u):
        batchsize = len(e_u)
        loss_real = F.sum(F.softplus(-r_u_real)) / batchsize
        loss_fake = F.sum(F.softplus( r_u_fake)) / batchsize
        loss = loss_fake + loss_real
        chainer.report({'loss'/ loss}, dis)
        return loss

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

        r_u_fake = gen(z, c_u)