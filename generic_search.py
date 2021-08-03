# follows: https://github.com/gaoxiang9430/sensei/blob/master/augment/ga_selector.py

import random
import copy

import torchvision.transforms.functional as TF


class Mutator(object):
    def __init__(
            self,
            flip=False, rotate=0, translate=0., translate_v=0., shear=0,
            zoom=1., brighten=1., contrast=1.):
        self.flip = flip
        self.rotate = rotate
        self.translate = translate
        self.translate_v = translate_v
        self.shear = shear
        self.zoom = zoom
        self.brighten = brighten
        self.contrast = contrast

    def __call__(self, x, idx):
        mx = []
        mx.append((TF.hflip(x) if self.flip else x, self._format_ops(idx, 'flip')))
        mx.append((TF.rotate(x, angle=self.rotate), self._format_ops(idx, 'rotate')))
        mx.append((TF.affine(x, 0, [self.translate, self.translate_v], 1., [0, 0]),
                   self._format_ops(idx, 'translate')))
        mx.append((TF.affine(x, 0, [0, 0], self.zoom, [0, 0]), self._format_ops(idx, 'zoom')))
        mx.append((TF.affine(x, 0, [0, 0], 1., self.shear), self._format_ops(idx, 'shear')))
        mx.append((TF.affine(x, self.rotate, [self.translate, self.translate_v], 1., [0, 0]),
                   self._format_ops(idx, ['rotate', 'translate'])))
        mx.append((TF.affine(x, self.rotate, [0, 0], 1., self.shear),
                   self._format_ops(idx, ['rotate', 'shear'])))
        mx.append((TF.adjust_brightness(x, self.brighten), self._format_ops(idx, 'brighten')))
        mx.append((TF.adjust_contrast(x, self.contrast), self._format_ops(idx, 'contrast')))
        return mx

    def _format_ops(self, idx, attrs):
        desc_img = 'image: {}'.format(idx)
        desc_mutate = []
        if not isinstance(attrs, list):
            attrs = [attrs]
        for attr in attrs:
            if attr in ('flip', 'shear'):
                desc_mutate.append(f'{attr}: {getattr(self, attr)}')
            elif attr == 'translate':
                desc_mutate.append(f"{attr}: ({getattr(self, attr):.3f}, {getattr(self, attr+'_v'):.3f})")
            elif attr in ('rotate', 'zoom'):
                desc_mutate.append(f'{attr}: {getattr(self, attr):.2f}')
            elif attr in ('brighten', 'contrast'):
                desc_mutate.append(f'{attr}: {getattr(self, attr):.3f}')
            else:
                raise ValueError('Invalid attribuate')
        desc_mutate = ', '.join(desc_mutate)
        desc = desc_img + ' | ' + desc_mutate
        return desc


class Chromosome(object):
    flip_range = [True, False]
    rotate_step = 0.25
    rotate_range = [r/4. for r in range(-60, 61)]  # [-15, 15, 0.25]
    translate_step = 0.005
    translate_range = [t/200. for t in range(-20, 21) ]  # [-10%, 10%, 0.5%]
    translate_v_step = 0.005
    translate_v_range = [t/200. for t in range(-20, 21) ]  # [-10%, 10%, 0.5%]
    shear_step = 1
    shear_range = range(-10, 11)  # [-10, 10, 1]

    zoom_step = 0.01
    zoom_range = [z/100. for z in range(90, 111)]  # [0.9, 1,1, 0.01]
    brighten_step = 0.025
    brighten_range = [b/40. for b in range(32, 49)]  # [0.8, 1.2, 0.025]
    contrast_step = 0.025
    contrast_range = [c/40. for c in range(32, 49)]  # [0.8, 1.2, 0.025]

    def __init__(
            self,
            mutator=Mutator(), converage=0,
            #  std=0,
            dst=0,
            rand_init=False, enable_filters=False):
        self.mutator = mutator
        self.cov = converage
        #  self.std = std
        self.dst = dst
        self.enable_filters = enable_filters
        if rand_init:
            self._random_init()

    def _random_init(self):
        self.mutator.flip = random.choice(self.flip_range)
        self.mutator.rotate = random.choice(self.rotate_range)
        self.mutator.translate = random.choice(self.translate_range)
        self.mutator.translate_v = random.choice(self.translate_range)
        self.mutator.shear = random.choice(self.shear_range)
        if self.enable_filters:
            self.mutator.zoom = random.choice(self.zoom_range)
            self.mutator.brighten = random.choice(self.brighten_range)
            self.mutator.contrast = random.choice(self.contrast_range)

    def __call__(self, x, idx):
        return self.mutator(x, idx)

    def mutate(self, nonsurprise):
        mutator_attr = list(self.mutator.__dict__.keys())
        if self.enable_filters:
            choices = random.sample(mutator_attr, 4)
        else:
            choices = random.sample(mutator_attr[:5], 3)
        op = random.choice(('add', 'substract', 'mirror'))
        m = copy.deepcopy(self.mutator)
        for c in choices:
            v = getattr(self.mutator, c)
            if c == 'flip':
                if nonsurprise is True:
                    v = random.choice([True, False])
                else:
                    v = not v  # revert it no matter what op
                setattr(m, c, v)
                continue
            r = eval('self.{}_range'.format(c))
            s = eval('self.{}_step'.format(c))
            t = random.choice(range(1, 4)) if nonsurprise is True else 1
            if op == 'add':
                v = min(max(v + t * s, r[0]), r[-1])
            elif op == 'substract':
                v = min(max(v - t * s, r[0]), r[-1])
            else:  # mirror
                if c in ('zoom', 'contrast', 'brighten'):
                    v = 2. - v
                else:
                    v *= -1
            setattr(m, c, v)
        return Chromosome(m, self.cov, enable_filters=self.enable_filters)

    def crossover(self, other):
        mutator_attr = self.mutator.__dict__.keys()
        if self.enable_filters:
            choice = random.choice(range(1, 254))  # 00000001 - 11111110
        else:
            choice = random.choice(range(1, 62))  # 0001- 111110
        m = copy.deepcopy(self.mutator)
        ids = '{0:08b}'.format(choice)
        for a, i in zip(mutator_attr, ids[::-1]):  # revert
            if i:
                v = getattr(other.mutator, a)
                setattr(m, a, v)
        return Chromosome(m, self.cov, enable_filters=self.enable_filters)

    def __sub__(self, other):
        a, b = self.mutator, other.mutator
        distance = 0
        distance += a.flip ^ b.flip
        distance += abs(a.rotate - b.rotate) / self.rotate_step
        distance += abs(a.translate - b.translate) / self.translate_step
        distance += abs(a.translate_v - b.translate_v) / self.translate_v_step
        distance += abs(a.shear - b.shear) / self.shear_step
        if self.enable_filters:
            distance += abs(a.zoom - b.zoom) / self.zoom_step
            distance += abs(a.brighten - b.brighten) / self.brighten_step
            distance += abs(a.contrast - b.contrast) / self.contrast_step
        return distance


class GenericSearcher(object):
    def __init__(self, opt, num_test):
        self.genesize = opt.genesize
        self.popsize = opt.popsize
        self.enable_filters = opt.enable_filters
        self.enable_iter = opt.enable_iter
        self.mutate_prob = opt.mutate_prob
        self.num_test = num_test
        self.mutator_seeds = self._init_mutator()
        self.seeds_nonsurprise = [False] * self.num_test
        self.history_pops = None
        self.mutator_pops = []

    def _init_mutator(self):
        e = self.enable_filters
        seeds = []
        for _ in range(self.num_test):
            mutators = []
            for _ in range(self.genesize):
                chromo = Chromosome(rand_init=True, enable_filters=e)
                mutators.append(chromo)
            seeds.append(mutators)
        return seeds

    def generate_next_population(self):
        prob = random.uniform(0, 1)
        if prob < self.mutate_prob:
            self.mutate()
        else:
            self.crossover()

        return self.mutator_pops

    def mutate(self):
        for i in range(self.num_test):
            q = self.mutator_seeds[i]
            pops = []
            for _ in range(self.popsize):
                chromo = random.choice(q)
                item = chromo.mutate(self.seeds_nonsurprise[i])
                pops.append(item)
            self.mutator_pops.append(pops)

    def crossover(self):
        for i in range(self.num_test):
            q = self.mutator_seeds[i]
            pops = []
            for _ in range(self.popsize):
                item1, item2 = random.sample(q, 2)
                item = item1.crossover(item2)
                pops.append(item)
            self.mutator_pops.append(pops)

    def fitness(self, covs=None):
        if covs is None:
            for i in range(self.num_test):
                q = self.mutator_seeds[i] + self.mutator_pops[i]
                m = random.sample(q, self.genesize)
                self.mutator_seeds[i] = m
        else:
            for i in range(self.num_test):
                for j in range(self.popsize):
                    self.mutator_pops[i][j].cov = covs[i][j].max().item()
                    #  self.mutator_pops[i][j].std = covs[i][j].std().item()
                    self.mutator_pops[i][j].dst = min([
                        self.mutator_pops[i][j] - m for m in self.history_pops[i]
                    ]) if self.history_pops is not None else 0
                least_cov = self.mutator_seeds[i][-1].cov
                q = self.mutator_seeds[i] + self.mutator_pops[i]
                random.shuffle(q)  # shuffle for zero converage layers
                if self.enable_iter:
                    #  q = sorted(q, key=lambda c: (int(c.cov*10), c.std), reverse=True)
                    q = sorted(q, key=lambda c: (int(c.dst+0.5), int(c.cov*10),), reverse=True)
                else:
                    q = sorted(q, key=lambda c: c.cov, reverse=True)
                self.mutator_seeds[i] = q
                new_least_cov = self.mutator_seeds[i][-1].cov
                self.seeds_nonsurprise[i] = not (new_least_cov > least_cov)
            self.history_pops = self.mutator_pops[:]

        self.mutator_pops.clear()

