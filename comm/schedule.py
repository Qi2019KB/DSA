# -*- coding: utf-8 -*-
import math
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


class ScheduleUtils:
    def __init__(self):
        pass

    @classmethod
    def get_cosine_schedule_with_warmup(cls, optimizer, num_warmup_steps, num_training_steps, num_cycles=7. / 16., last_epoch=-1):
        def _lr_lambda(current_step):
            if current_step < num_warmup_steps: return float(current_step) / float(max(1, num_warmup_steps))
            no_progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0., math.cos(math.pi * num_cycles * no_progress))

        return LambdaLR(optimizer, _lr_lambda, last_epoch)

    @classmethod
    def count_thr_update(cls, args):
        if args.count_thr_type == 'Increase':
            return int(cls.get_lambda_with_line_increase(args.epo, args.count_thr_max, args.count_thr_min, args.count_thr_rampup))
        elif args.count_thr_type == 'Decrease':
            return int(cls.get_lambda_with_line_decrease(args.epo, args.count_thr_max, args.count_thr_min, args.count_thr_rampup))
        else:
            return args.count_thr

    @classmethod
    def score_thr_update(cls, args):
        if args.score_thr_type == 'Increase':
            return cls.get_lambda_with_line_increase(args.epo, args.score_thr_max, args.score_thr_min, args.score_thr_rampup)
        elif args.score_thr_type == 'Decrease':
            return cls.get_lambda_with_line_decrease(args.epo, args.score_thr_max, args.score_thr_min, args.score_thr_rampup)
        else:
            return args.score_thr

    @classmethod
    def taut_alpha_update(cls, args):
        if args.taut_alpha_type == 'Increase':
            return cls.get_lambda_with_line_increase(args.epo, args.taut_alpha_max, args.taut_alpha_min, args.taut_alpha_rampup)
        elif args.taut_alpha_type == 'Decrease':
            return cls.get_lambda_with_line_decrease(args.epo, args.taut_alpha_max, args.taut_alpha_min, args.taut_alpha_rampup)
        return args.taut_alpha

    @classmethod
    def get_lambda_with_sigmoid_increase(cls, epo, max_value, min_value, rampup_value):
        return cls._value_increase(cls, epo, max_value, min_value, rampup_value)

    @classmethod
    def get_lambda_with_sigmoid_decrease(cls, epo, max_value, min_value, rampup_value):
        return cls._value_decrease(cls, epo, max_value, min_value, rampup_value)

    @classmethod
    def get_lambda_with_line_increase(cls, epo, max_value, min_value, rampup_value):
        return min_value + (max_value-min_value)*(min((epo+1)/rampup_value, 1))

    @classmethod
    def get_lambda_with_line_decrease(cls, epo, max_value, min_value, rampup_value):
        return max_value - (max_value-min_value)*(min((epo+1)/rampup_value, 1))

    def _value_increase(self, epo, maxValue, minValue, rampup):
        return minValue + (maxValue - minValue) * self._sigmoid_rampup(self, epo, rampup)

    def _value_decrease(self, epo, maxValue, minValue, rampup):
        return minValue + (maxValue - minValue) * (1.0 - self._sigmoid_rampup(self, epo, rampup))

    def _sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))