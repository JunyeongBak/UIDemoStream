import sys
import torch
import torch.nn as nn
import math
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar
)
from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtGui import QFont

# pip install torch
# pip install pyside6
# --- 1. 완전한 트랜스포머 모델 (인코더-디코더 구조) ---

class PositionalEncoding(nn.Module):
    # (이전 코드와 동일)
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_tokens, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.d_model = d_model
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.token_embedding = nn.Embedding(num_tokens, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.generator = nn.Linear(d_model, num_tokens)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.token_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.token_embedding(tgt) * math.sqrt(self.d_model))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        src_emb = self.positional_encoding(self.token_embedding(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_emb, src_mask)

    def decode(self, tgt, memory, tgt_mask):
        tgt_emb = self.positional_encoding(self.token_embedding(tgt) * math.sqrt(self.d_model))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)

# --- 2. 학습 및 추론 로직 ---

class TrainingWorker(QObject):
    progress_updated = Signal(int)
    training_finished = Signal(str)

    def __init__(self, model, vocab):
        super().__init__()
        self.model = model
        self.vocab = vocab

    def run_training(self):
        try:
            self.model.train()
            src_text = "hello"
            tgt_text = "world"
            BOS_TOKEN, EOS_TOKEN, PAD_TOKEN = self.vocab['<s>'], self.vocab['</s>'], self.vocab['<pad>']
            src_ids = torch.LongTensor([[BOS_TOKEN] + [self.vocab[c] for c in src_text] + [EOS_TOKEN]])
            tgt_ids = torch.LongTensor([[BOS_TOKEN] + [self.vocab[c] for c in tgt_text] + [EOS_TOKEN]])
            tgt_input = tgt_ids[:, :-1]
            tgt_output = tgt_ids[:, 1:]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1))
            src_mask = torch.zeros((src_ids.shape[1], src_ids.shape[1]), dtype=torch.bool)
            criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            epochs = 300
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = self.model(src_ids, tgt_input, src_mask, tgt_mask, None, None, None)
                loss = criterion(output.view(-1, len(self.vocab)), tgt_output.reshape(-1))
                loss.backward()
                optimizer.step()
                if (epoch + 1) % (epochs // 10) == 0:
                    self.progress_updated.emit(int((epoch + 1) / epochs * 100))
            self.model.eval()
            self.training_finished.emit(f"'hello' -> 'world' 생성 학습 완료! (총 {epochs}회 반복)")
        except Exception as e:
            self.training_finished.emit(f"학습 중 오류 발생:\n\n{e}")

# --- 3. PySide6 GUI 애플리케이션 ---

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        # 단어 사전 (Vocabulary) 정의
        self.SRC_VOCAB = "abcdefghijklmnopqrstuvwxyz "
        self.TGT_VOCAB = "abcdefghijklmnopqrstuvwxyz "
        special_symbols = ['<pad>', '<s>', '</s>']
        self.vocab = {symbol: i for i, symbol in enumerate(special_symbols)}
        for char in sorted(list(set(self.SRC_VOCAB + self.TGT_VOCAB))):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

        # 모델 초기화
        self.d_model = 128
        self.model = Seq2SeqTransformer(
            num_tokens=len(self.vocab), d_model=self.d_model, nhead=4,
            num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512
        )
        self.model.eval()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Seq2Seq 트랜스포머 생성기")
        self.setGeometry(300, 300, 600, 500)
        self.layout = QVBoxLayout()
        self.setFont(QFont())

        self.layout.addWidget(QLabel("생성할 문장의 시작 단어(Seed)를 입력하세요:"))
        self.input_text = QLineEdit(placeholderText="예: hello")
        self.analyze_button = QPushButton("생성 실행")
        self.train_button = QPushButton("간단한 학습 시작 ('hello' -> 'world')")
        self.progress_bar = QProgressBar()
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        
        self.layout.addWidget(self.input_text)
        self.layout.addWidget(self.analyze_button)
        self.layout.addWidget(self.train_button)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(QLabel("--- 결과 ---"))
        self.layout.addWidget(self.result_display)
        
        self.setLayout(self.layout)
        
        self.analyze_button.clicked.connect(self.generate_text)
        self.input_text.returnPressed.connect(self.generate_text)
        self.train_button.clicked.connect(self.start_training)

    def generate_text(self):
        text = self.input_text.text().lower()
        if not text:
            self.result_display.setText("생성을 시작할 단어를 먼저 입력해주세요.")
            return

        try:
            self.model.eval()
            BOS_TOKEN, EOS_TOKEN = self.vocab['<s>'], self.vocab['</s>']
            src_ids = torch.LongTensor([[BOS_TOKEN] + [self.vocab.get(c, 0) for c in text] + [EOS_TOKEN]])
            src_mask = (torch.zeros(src_ids.shape[1], src_ids.shape[1])).type(torch.bool)
            memory = self.model.encode(src_ids, src_mask)
            ys = torch.ones(1, 1).fill_(BOS_TOKEN).type(torch.long)
            generated_sequence = []
            
            for _ in range(20):
                tgt_mask = (nn.Transformer.generate_square_subsequent_mask(ys.size(1))).type(torch.bool)
                out = self.model.decode(ys, memory, tgt_mask)
                prob = self.model.generator(out[:, -1, :])
                _, next_word_idx = torch.max(prob, dim=1)
                next_word_idx = next_word_idx.item()
                ys = torch.cat([ys, torch.ones(1, 1).type_as(src_ids.data).fill_(next_word_idx)], dim=1)
                if next_word_idx == EOS_TOKEN:
                    break
                generated_sequence.append(self.rev_vocab.get(next_word_idx, ''))

            result_text = f"입력: '{text}'\n"
            result_text += f"생성 결과: '{''.join(generated_sequence)}'"
        
        except Exception as e:
            result_text = f"생성 중 오류 발생:\n\n{e}"

        self.result_display.setText(result_text)

    def start_training(self):
        self.train_button.setEnabled(False)
        self.result_display.setText("학습을 시작합니다...")
        self.progress_bar.setValue(0)
        
        self.thread = QThread()
        self.worker = TrainingWorker(self.model, self.vocab)
        self.worker.moveToThread(self.thread)
        
        # 시그널 연결
        self.thread.started.connect(self.worker.run_training)
        self.worker.training_finished.connect(self.on_training_finished)
        self.worker.progress_updated.connect(lambda v: self.progress_bar.setValue(v))
        
        # [수정된 부분] 스레드 종료 시 worker와 thread 객체를 모두 안전하게 삭제하도록 연결합니다.
        # 이것이 스레드 부모-자식 관계 충돌 경고를 해결해줍니다.
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()

    def on_training_finished(self, message):
        self.result_display.setText(message)
        self.train_button.setEnabled(True)
        # 스레드가 아직 실행 중일 경우에만 종료 신호를 보냅니다.
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.quit()

# --- 4. 애플리케이션 실행 ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
