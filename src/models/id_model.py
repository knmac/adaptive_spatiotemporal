from .base_model import BaseModel


class IdModel(BaseModel):
    def __init__(self, device, modality):
        super(IdModel, self).__init__(device)
        self.modality = modality

    def forward(self, input):
        return input
