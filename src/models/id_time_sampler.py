from .base_model import BaseModel


class IdTimeSampler(BaseModel):
    def __init__(self, device, modality):
        super(IdTimeSampler, self).__init__(device)
        self.modality = modality

    def forward(self, input, heavy_model):
        return input
