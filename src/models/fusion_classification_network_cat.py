from .fusion_classification_network import Fusion_Classification_Network


class Fusion_Classification_Network_Cat(Fusion_Classification_Network):
    """Wrapper of Fusion_Classification_Network with cat fusion only.
    The input for forward function is already concatenated
    """
    def __init__(self, device, feature_dim, modality, num_class, num_segments,
                 consensus_type, before_softmax, dropout):
        # Initialize with fixed value of midfusion
        super(Fusion_Classification_Network_Cat, self).__init__(
            feature_dim=feature_dim, modality=modality, midfusion='concat',
            num_class=num_class, consensus_type=consensus_type,
            before_softmax=before_softmax, dropout=dropout,
            num_segments=num_segments)
        self.device = device

    def forward(self, inputs):
        """Wrapper of the parent forward because we assume the inputs here are
        already concatenated
        """
        if len(self.modality) > 1:  # Fusion
            base_out = self.fc1(inputs)
            base_out = self.relu(base_out)
        else:  # Single modality
            base_out = inputs

        # ---------------------------------------------------------------------
        # This part below is the same as the parent class
        if self.dropout > 0:
            base_out = self.dropout_layer(base_out)

        # Snippet-level predictions and temporal aggregation with consensus
        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            # Verb
            base_out_verb = self.fc_verb(base_out)
            if not self.before_softmax:
                base_out_verb = self.softmax(base_out_verb)
            if self.reshape:
                base_out_verb = base_out_verb.view((-1, self.num_segments) + base_out_verb.size()[1:])
            output_verb = self.consensus(base_out_verb)

            # Noun
            base_out_noun = self.fc_noun(base_out)
            if not self.before_softmax:
                base_out_noun = self.softmax(base_out_noun)
            if self.reshape:
                base_out_noun = base_out_noun.view((-1, self.num_segments) + base_out_noun.size()[1:])
            output_noun = self.consensus(base_out_noun)

            output = (output_verb.squeeze(1), output_noun.squeeze(1))

        else:
            base_out = self.fc_action(base_out)
            if not self.before_softmax:
                base_out = self.softmax(base_out)
            if self.reshape:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

            output = self.consensus(base_out)
            output = output.squeeze(1)

        return output
