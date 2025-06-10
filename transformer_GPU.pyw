import torch
import torch.nn as nn
import math
import copy

# 이 코드는 PyTorch를 사용하여 Transformer 모델의 아키텍처를 처음부터 구현합니다.
# 각 구성 요소(Multi-Head Attention, Positional Encoding 등)를 개별 클래스로 만들어
# 전체 구조를 쉽게 이해할 수 있도록 하는 데 목적이 있습니다.

class Transformer(nn.Module):
    """
    완전한 Transformer 모델 클래스. Encoder와 Decoder를 포함합니다.
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 src_vocab_size: int = 1000, tgt_vocab_size: int = 1000):
        super(Transformer, self).__init__()

        # --- 인코더(Encoder) 부분 ---
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # --- 디코더(Decoder) 부분 ---
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # --- 최종 출력 레이어 ---
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model
        self._reset_parameters()

    def _reset_parameters(self):
        """파라미터 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None,
                src_padding_mask: torch.Tensor = None, tgt_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the Transformer model.
        Args:
            src: 소스 시퀀스 (batch_size, src_seq_len)
            tgt: 타겟 시퀀스 (batch_size, tgt_seq_len)
            src_mask, tgt_mask: 어텐션 마스크
            ..._padding_mask: 패딩 마스크
        Returns:
            출력 텐서 (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # 1. 인코더 포워드 패스
        # 소스 문장을 임베딩하고 위치 정보를 더합니다.
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        # 인코더에 통과시켜 메모리(컨텍스트 벡터)를 생성합니다.
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)

        # 2. 디코더 포워드 패스
        # 타겟 문장을 임베딩하고 위치 정보를 더합니다.
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        # 인코더의 출력(memory)과 타겟 임베딩을 디코더에 통과시킵니다.
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                          tgt_padding_mask, memory_key_padding_mask)
        
        # 3. 최종 출력
        # 디코더의 출력을 선형 레이어에 통과시켜 어휘 크기의 로짓(logits)으로 변환합니다.
        return self.fc_out(output)

class PositionalEncoding(nn.Module):
    """
    위치 정보를 임베딩 벡터에 추가합니다.
    Transformer는 순서 정보를 모르기 때문에, 각 단어의 위치 정보를 알려줘야 합니다.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        
        # 짝수 차원에는 sin 함수, 홀수 차원에는 cos 함수를 적용합니다.
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # pe를 버퍼로 등록합니다. 이 텐서는 모델의 파라미터는 아니지만, state_dict에 저장됩니다.
        self.register_buffer('pe', pe.transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 임베딩 텐서 (batch_size, seq_len, d_model)
        """
        # 입력 텐서 x에 위치 인코딩을 더합니다.
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """
    디코더의 셀프 어텐션에서 미래 토큰을 참조하지 못하도록 마스크를 생성합니다.
    예를 들어, 세 번째 단어를 예측할 때 첫 번째와 두 번째 단어만 참고해야 합니다.
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)

def main():
    # --- 0. 장치 설정 (GPU 우선 사용) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 하이퍼파라미터 정의 ---
    SRC_VOCAB_SIZE = 2000 # 소스 언어의 어휘 크기
    TGT_VOCAB_SIZE = 2000 # 타겟 언어의 어휘 크기
    D_MODEL = 512          # 모델의 임베딩 차원 (논문과 동일)
    NHEAD = 8              # 멀티헤드 어텐션의 헤드 수 (논문과 동일)
    NUM_ENCODER_LAYERS = 3 # 인코더 레이어 수 (실습을 위해 줄임)
    NUM_DECODER_LAYERS = 3 # 디코더 레이어 수 (실습을 위해 줄임)
    DIM_FEEDFORWARD = 2048 # 피드포워드 신경망의 차원 (논문과 동일)
    DROPOUT = 0.1          # 드롭아웃 비율
    
    # --- 2. 모델 인스턴스화 ---
    model = Transformer(d_model=D_MODEL, nhead=NHEAD,
                        num_encoder_layers=NUM_ENCODER_LAYERS,
                        num_decoder_layers=NUM_DECODER_LAYERS,
                        dim_feedforward=DIM_FEEDFORWARD,
                        dropout=DROPOUT,
                        src_vocab_size=SRC_VOCAB_SIZE,
                        tgt_vocab_size=TGT_VOCAB_SIZE).to(device)
    
    # --- 3. 간단한 더미 데이터 생성 ---
    # 실제로는 토크나이저를 통해 생성된 정수 시퀀스입니다.
    BATCH_SIZE = 4
    SRC_SEQ_LEN = 10
    TGT_SEQ_LEN = 12

    src_data = torch.randint(1, SRC_VOCAB_SIZE, (BATCH_SIZE, SRC_SEQ_LEN)).to(device)
    tgt_data = torch.randint(1, TGT_VOCAB_SIZE, (BATCH_SIZE, TGT_SEQ_LEN)).to(device)
    
    print("\n--- Input Data Shapes ---")
    print(f"Source data shape: {src_data.shape}")
    print(f"Target data shape: {tgt_data.shape}")

    # --- 4. 마스크 생성 ---
    # 디코더가 미래 시점의 단어를 보지 못하게 막는 마스크
    tgt_mask = generate_square_subsequent_mask(TGT_SEQ_LEN, device)
    
    # 패딩 마스크 (여기서는 모든 시퀀스 길이가 같으므로 None으로 처리)
    src_padding_mask = None # (torch.zeros(src.shape)).bool().to(device)
    tgt_padding_mask = None # (torch.zeros(tgt.shape)).bool().to(device)

    # --- 5. 손실 함수와 옵티마이저 정의 ---
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 패딩 토큰(0)은 손실 계산에서 제외
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # --- 6. 간단한 학습 루프 ---
    print("\n--- Start Simple Training Loop ---")
    model.train() # 모델을 학습 모드로 설정

    for epoch in range(5):
        optimizer.zero_grad()
        
        # 모델 포워드 패스
        output = model(src=src_data, tgt=tgt_data, tgt_mask=tgt_mask, 
                       src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
        
        # 출력 (Batch, SeqLen, VocabSize) -> (Batch*SeqLen, VocabSize)
        # 타겟 (Batch, SeqLen) -> (Batch*SeqLen)
        # 손실을 계산하기 위해 텐서 모양을 조정합니다.
        loss = criterion(output.view(-1, TGT_VOCAB_SIZE), tgt_data.view(-1))
        
        # 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()
        
        print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

    print("\n--- Training Finished ---")
    
    # --- 7. 추론 (Inference) 예시 ---
    model.eval() # 모델을 평가 모드로 설정
    with torch.no_grad():
        # 추론 시에는 디코더에 한 단어씩 입력을 주며 다음 단어를 예측합니다.
        # 여기서는 간단히 학습된 모델에 데이터를 통과시켜 출력 형태만 확인합니다.
        eval_output = model(src=src_data, tgt=tgt_data, tgt_mask=tgt_mask)
        print("\n--- Inference Output Shape ---")
        print(f"Output shape: {eval_output.shape}")
        
        # 가장 확률이 높은 단어의 인덱스를 예측값으로 선택
        predictions = eval_output.argmax(dim=-1)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Sample prediction for first batch: \n{predictions[0]}")


if __name__ == '__main__':
    main()
