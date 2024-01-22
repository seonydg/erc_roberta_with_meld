import torch
import csv
from transformers import RobertaTokenizer
from torch.utils.data import Dataset


def data_split(session):
    final_data = []
    split_session = []

    for line in session:
        split_session.append(line)
        final_data.append(split_session[:])

    return final_data

class MyDataset(Dataset):
    def __init__(self, data_path):
        f = open(data_path, 'r')
        rdr = csv.reader(f)

        """ 추가 """

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        """ 세션 데이터 저장할 것"""
        self.session_dataset = []
        session = []
        speaker_set = []

        """ 실제 데이터 저장 방식 """
        pre_sess = 'start'
        for i, line in enumerate(rdr):
            if i == 0:
                """ 저장할 데이터들 index 확인 """
                header  = line
                utt_idx = header.index('Utterance')
                speaker_idx = header.index('Speaker')
                emo_idx = header.index('Emotion')
                sess_idx = header.index('Dialogue_ID')
            else:
                utt = line[utt_idx]
                speaker = line[speaker_idx]
                """ 유니크한 스피커로 바꾸기 """
                if speaker in speaker_set:
                    uniq_speaker = speaker_set.index(speaker)
                else:
                    speaker_set.append(speaker)
                    uniq_speaker = speaker_set.index(speaker)
                emotion = line[emo_idx]
                sess = line[sess_idx]

                if pre_sess == 'start' or sess == pre_sess:
                    session.append([uniq_speaker, utt, emotion])
                else:
                    """ 세션 데이터 저장 """
                    self.session_dataset += data_split(session)
                    session = [[uniq_speaker, utt, emotion]]
                    speaker_set = []

                pre_sess = sess
        """ 마지막 세션 저장 """
        self.session_dataset += data_split(session)

        self.emoList = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        f.close()

    def __len__(self): # 기본적인 구성
        return len(self.session_dataset)

    def __getitem__(self, idx): # 기본적인 구성
        return self.session_dataset[idx]

    def padding(self, batch_input_token): # batch size가 1보다 클 경우 추가
        """ 512 토큰 길이 넘으면 잘라내기 """
        batch_token_ids = batch_input_token['input_ids']
        batch_attention_masks = batch_input_token['attention_mask']

        trunc_batch_token_ids = []
        trunc_batch_attention_masks = []

        for batch_token_id, batch_attention_mask in zip(batch_token_ids, batch_attention_masks):
            if len(batch_token_id) > self.tokenizer.model_max_length:
                trunc_batch_token_id = [batch_token_id[0]] + batch_token_id[1:][-self.tokenizer.model_max_length+1:]
                trunc_batch_attention_mask = [batch_attention_mask[0]] + batch_attention_mask[1:][-self.tokenizer.model_max_length+1:]

                trunc_batch_token_ids.append(trunc_batch_token_id)
                trunc_batch_attention_masks.append(trunc_batch_attention_mask)
            else:
                trunc_batch_token_ids.append(batch_token_id)
                trunc_batch_attention_masks.append(batch_attention_mask)

        """ padding token으로 패딩하기 """
        # [10, 30, 50] -> [50, 50, 50]
        # 50-10=40 , 50-30=20 : 패딩토큰으로 채운다. <pad>
        max_length = max([len(x) for x in trunc_batch_token_ids])
        padding_tokens = []
        padding_attention_masks = []
        for batch_token_id, batch_attention_mask in zip(trunc_batch_token_ids, trunc_batch_attention_masks):
            padding_tokens.append(batch_token_id + [self.tokenizer.pad_token_id for _ in range(max_length - len(batch_token_id))])
            padding_attention_masks.append(batch_attention_mask + [0 for _ in range(max_length - len(batch_token_id))])

        return torch.tensor(padding_tokens), torch.tensor(padding_attention_masks)

    def collate_fn(self, sessions): # 배치를 위한 구성
        '''
            input:
                data: [(session1), (session2), ... ]
            return:
                batch_input_tokens_pad: (B, L) padded
                batch_labels: (B)
        '''
        ## [발화1, 발화2, ..., 발화8]
        # 발화1~발화7 컨텍스트로 사용한다면 입력이 길어진다.
        # 발화1 같은 경우는 발화8에 덜중요할거에요.
        # 적절하게 컨텍스트 길이를 조절해도된다.
        # 3개로 정한다면, [발화5,발화6,발화7,발화8]
        """ 추가 """
        batch_input, batch_labels = [], []
        batch_PM_input = []
        for session in sessions:
            input_str = self.tokenizer.cls_token

            """ For PM """
            current_speaker, current_utt, current_emotion = session[-1]
            PM_input = []
            for i, line in enumerate(session):
                speaker, utt, emotion = line
                input_str += " " + utt + self.tokenizer.sep_token
                if i < len(session)-1 and current_speaker == speaker:
                    PM_input.append(self.tokenizer.encode(utt, add_special_tokens=True, return_tensors='pt'))
                    # [cls_token, tokens, sep_token]

            """ For CoM """
            batch_input.append(input_str)
            batch_labels.append(self.emoList.index(emotion))
            batch_PM_input.append(PM_input)
        batch_input_token = self.tokenizer(batch_input, add_special_tokens=False)
        batch_padding_token, batch_padding_attention_mask = self.padding(batch_input_token)

        return batch_padding_token, batch_padding_attention_mask, batch_PM_input, torch.tensor(batch_labels)