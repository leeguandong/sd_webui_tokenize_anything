# ------------------------------------------------------------------------
# Copyright (c) 2023-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, esither express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Learning rate schedulers."""

import math


class ConstantLR(object):
    """Constant LR scheduler."""

    def __init__(self, **kwargs):
        self._lr_max = kwargs.pop("lr_max")
        self._lr_min = kwargs.pop("lr_min", 0)
        self._warmup_steps = kwargs.pop("warmup_steps", 0)
        self._warmup_factor = kwargs.pop("warmup_factor", 0)
        if kwargs:
            raise ValueError("Unexpected arguments: " + ",".join(v for v in kwargs))
        self._step_count = 0
        self._last_decay = 1.0

    def step(self):
        self._step_count += 1

    def get_lr(self):
        if self._step_count < self._warmup_steps:
            alpha = (self._step_count + 1.0) / self._warmup_steps
            return self._lr_max * (alpha + (1.0 - alpha) * self._warmup_factor)
        return self._lr_min + (self._lr_max - self._lr_min) * self.get_decay()

    def get_decay(self):
        return self._last_decay


class CosineLR(ConstantLR):
    """LR scheduler with cosine decay."""

    def __init__(self, lr_max, max_steps, lr_min=0, decay_step=1, **kwargs):
        super(CosineLR, self).__init__(lr_max=lr_max, lr_min=lr_min, **kwargs)
        self._decay_step = decay_step
        self._max_steps = max_steps

    def get_decay(self):
        t = self._step_count - self._warmup_steps
        t_max = self._max_steps - self._warmup_steps
        if t > 0 and t % self._decay_step == 0:
            self._last_decay = 0.5 * (1.0 + math.cos(math.pi * t / t_max))
        return self._last_decay


class LinearLR(ConstantLR):
    """LR scheduler with linear decay."""

    def __init__(self, lr_max, max_steps, lr_min=0, decay_step=1, **kwargs):
        super(LinearLR, self).__init__(lr_max=lr_max, lr_min=lr_min, **kwargs)
        self._decay_step = decay_step
        self._max_steps = max_steps

    def get_decay(self):
        t = self._step_count - self._warmup_steps
        t_max = self._max_steps - self._warmup_steps
        if t > 0 and t % self._decay_step == 0:
            self._last_decay = 1.0 - float(t) / t_max
        return self._last_decay
