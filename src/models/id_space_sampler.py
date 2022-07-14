from .base_model import BaseModel


class IdSpaceSampler(BaseModel):
    def __init__(self, device, modality):
        super(IdSpaceSampler, self).__init__(device)
        self.modality = modality

    def forward(self, input):
        return input
