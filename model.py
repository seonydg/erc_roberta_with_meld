import torch
import torch.nn as nn
from transformers import RobertaModel

class ERC_model(nn.Module):
    def __init__(self, clsNum):
        super(ERC_model, self).__init__()

        self.com_model = RobertaModel.from_pretrained('roberta-base')
        self.pm_model = RobertaModel.from_pretrained('roberta-base')

        self.hiddenDim = self.com_model.config.hidden_size
        zero = torch.empty(2, 1, self.hiddenDim)
        self.h0 = torch.zeros_like(zero).cuda()
        self.speakerGRU = nn.GRU(self.hiddenDim, self.hiddenDim, 2, dropout=0.3)
        self.W = nn.Linear(self.hiddenDim, clsNum)

    def forward(self, batch_padding_token, batch_padding_attention_mask, batch_PM_input):
        # CoM
        batch_com_out = self.com_model(input_ids=batch_padding_token, attention_mask=batch_padding_attention_mask)['last_hidden_state']
        batch_com_final = batch_com_out[:, 0, :]

        # PM
        batch_pm_gru_final = []
        for PM_inputs in batch_PM_input:
            if PM_inputs:
                pm_outs = []
                for PM_input in PM_inputs:
                    pm_out = self.pm_model(PM_input)['last_hidden_state'][:, 0, :]
                    pm_outs.append(pm_out)

                pm_outs = torch.cat(pm_outs, 0).unsqueeze(1)
                pm_gru_outs, _ = self.speakerGRU(pm_outs, self.h0)
                pm_gru_final = pm_gru_outs[-1, :, :]
                batch_pm_gru_final.append(pm_gru_final)
            else:
                batch_pm_gru_final.append(torch.zeros(1, self.hiddenDim).cuda())

        batch_pm_gru_final = torch.cat(batch_pm_gru_final, 0)

        # score matrix
        final_out = self.W(batch_com_final + batch_pm_gru_final)

        return final_out