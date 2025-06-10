import sys
import torch
import torch.nn as nn
import math
import random
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar, QFileDialog
)
from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtGui import QFont

# pip install pyside6
# GPU 설치: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

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

    def __init__(self, model, vocab, device):
        super().__init__()
        self.model = model
        self.vocab = vocab
        self.device = device

    def run_training(self):
        try:
            self.model.train()
            # [수정] 한글 학습 데이터를 추가합니다.
            training_data = [
                ("hello", "world"),
                ("good bye", "see you"),
                ("how are you", "i am fine"),
                ("what is your name", "i am a model"),
                ("사랑해", "나도 사랑해"),
                ("안녕", "안녕하세요"),
                ("소정", "사랑해"),
                ("강소정", "사랑해")
            ]
            BOS_TOKEN, EOS_TOKEN, PAD_TOKEN = self.vocab['<s>'], self.vocab['</s>'], self.vocab['<pad>']
            criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            epochs = 500
            for epoch in range(epochs):
                src_text, tgt_text = random.choice(training_data)
                src_ids = torch.LongTensor([[BOS_TOKEN] + [self.vocab[c] for c in src_text] + [EOS_TOKEN]]).to(self.device)
                tgt_ids = torch.LongTensor([[BOS_TOKEN] + [self.vocab[c] for c in tgt_text] + [EOS_TOKEN]]).to(self.device)
                tgt_input = tgt_ids[:, :-1]
                tgt_output = tgt_ids[:, 1:]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)
                src_mask = torch.zeros((src_ids.shape[1], src_ids.shape[1]), device=self.device).type(torch.bool)
                optimizer.zero_grad()
                output = self.model(src_ids, tgt_input, src_mask, tgt_mask, None, None, None)
                loss = criterion(output.view(-1, len(self.vocab)), tgt_output.reshape(-1))
                loss.backward()
                optimizer.step()
                if (epoch + 1) % (epochs // 10) == 0:
                    self.progress_updated.emit(int((epoch + 1) / epochs * 100))
            self.model.eval()
            self.training_finished.emit(f"다양한 데이터로 학습 완료! (총 {epochs}회 반복)")
        except Exception as e:
            self.training_finished.emit(f"학습 중 오류 발생:\n\n{e}")

# --- 3. PySide6 GUI 애플리케이션 ---

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_vocab()
        self.init_model()
        self.init_ui()

    # 이 부분을 SentencePiece 등의 토크나이저로 바꿔야 한다.
    def init_vocab(self):
        # [수정] 단어 사전에 한글을 추가합니다.
        english_chars = "abcdefghijklmnopqrstuvwxyz "
        korean_chars = "가각간갇갈감갑갓강갖갗갘같갚갛나다라락란랃랄람랍랏랑랒랓랔랕랖랗마바사아악안앋알암압앗앙앚앛앜앝앞앟자차카타파하개내대래매배새애재채캐태패해갸냐댜랴먀뱌샤야쟈챠캬탸퍄햐거너더러머버서어저적전젇절점접젓정젖젗젘젙젚젛처커터퍼허게네데레메베세에제체케테페헤겨녀녁년녇녈념녑녓녕녖녗녘녙녚녛뎌려며벼셔여져쳐켜텨펴혀고노도로모보소오조초코토포호교뇨됴료묘뵤쇼요죠쵸쿄툐표효구누두루무부수우주추쿠투푸후규뉴듀류뮤뷰슈유쥬츄큐튜퓨휴그느드르므브스으즈츠크트프흐기니디리미비시이지치키티피히" # 학습 데이터에 있는 모든 한글 문자
        
        self.SRC_VOCAB = english_chars + "".join(sorted(list(set(korean_chars))))
        self.TGT_VOCAB = self.SRC_VOCAB

        special_symbols = ['<pad>', '<s>', '</s>']
        self.vocab = {symbol: i for i, symbol in enumerate(special_symbols)}
        for char in self.SRC_VOCAB:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

    def init_model(self):
        self.d_model = 128
        self.model = Seq2SeqTransformer(
            num_tokens=len(self.vocab), d_model=self.d_model, nhead=4,
            num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512
        )
        self.model.to(self.device)
        self.model.eval()

    def init_ui(self):
        self.setWindowTitle("Seq2Seq 트랜스포머 생성기 (한글 지원)")
        self.setGeometry(300, 300, 600, 550)
        self.layout = QVBoxLayout()
        self.setFont(QFont())

        self.layout.addWidget(QLabel("생성할 문장의 시작 단어(Seed)를 입력하세요:"))
        self.input_text = QLineEdit(placeholderText="예: hello, 사랑해 등")
        self.analyze_button = QPushButton("생성 실행")
        self.train_button = QPushButton("간단한 학습 시작")
        self.reset_button = QPushButton("모델 초기화 (가중치 리셋)")
        self.save_button = QPushButton("모델 저장 (.pkl)")
        self.load_button = QPushButton("모델 불러오기 (.pkl)")
        self.save_button.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        
        self.layout.addWidget(self.input_text)
        self.layout.addWidget(self.analyze_button)
        self.layout.addWidget(self.train_button)
        self.layout.addWidget(self.reset_button)
        self.layout.addWidget(self.save_button)
        self.layout.addWidget(self.load_button)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(QLabel("--- 결과 ---"))
        self.layout.addWidget(self.result_display)
        
        device_info_label = QLabel(f"연산 장치: {str(self.device).upper()}")
        device_info_label.setAlignment(Qt.AlignRight)
        self.layout.addWidget(device_info_label)
        
        self.setLayout(self.layout)
        
        self.analyze_button.clicked.connect(self.generate_text)
        self.input_text.returnPressed.connect(self.generate_text)
        self.train_button.clicked.connect(self.start_training)
        self.reset_button.clicked.connect(self.reset_model)
        self.save_button.clicked.connect(self.save_model)
        self.load_button.clicked.connect(self.load_model)

    def generate_text(self):
        # [수정] 입력 텍스트를 소문자로 바꾸는 .lower() 부분을 제거하여 한글 입력을 그대로 받습니다.
        text = self.input_text.text()
        if not text:
            self.result_display.setText("생성을 시작할 단어를 먼저 입력해주세요.")
            return

        try:
            self.model.eval()
            BOS_TOKEN, EOS_TOKEN = self.vocab['<s>'], self.vocab['</s>']
            # .lower()를 제거하여 한글이 올바르게 처리되도록 합니다.
            src_ids = torch.LongTensor([[BOS_TOKEN] + [self.vocab.get(c, 0) for c in text] + [EOS_TOKEN]]).to(self.device)
            src_mask = (torch.zeros(src_ids.shape[1], src_ids.shape[1])).type(torch.bool).to(self.device)
            memory = self.model.encode(src_ids, src_mask)
            ys = torch.ones(1, 1).fill_(BOS_TOKEN).type(torch.long).to(self.device)
            generated_sequence = []
            
            for _ in range(30):
                tgt_mask = (nn.Transformer.generate_square_subsequent_mask(ys.size(1))).type(torch.bool).to(self.device)
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
        self.reset_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.result_display.setText("학습을 시작합니다...")
        self.progress_bar.setValue(0)
        
        self.thread = QThread()
        self.worker = TrainingWorker(self.model, self.vocab, self.device)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run_training)
        self.worker.training_finished.connect(self.on_training_finished)
        self.worker.progress_updated.connect(lambda v: self.progress_bar.setValue(v))
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()

    def on_training_finished(self, message):
        self.result_display.setText(message)
        self.train_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.save_button.setEnabled(True)
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.quit()
    
    def reset_model(self):
        self.result_display.setText("모델의 가중치를 초기 상태로 리셋했습니다.")
        self.init_model()
        self.progress_bar.setValue(0)
        self.save_button.setEnabled(False)

    def save_model(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "모델 저장", "", "Pickle Files (*.pkl)")
        if file_path:
            try:
                torch.save(self.model.state_dict(), file_path)
                self.result_display.setText(f"모델이 다음 경로에 저장되었습니다:\n{file_path}")
            except Exception as e:
                self.result_display.setText(f"모델 저장 중 오류 발생:\n\n{e}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "모델 불러오기", "", "Pickle Files (*.pkl)")
        if file_path:
            try:
                self.model.load_state_dict(torch.load(file_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.save_button.setEnabled(True)
                self.result_display.setText(f"다음 경로에서 모델을 불러왔습니다:\n{file_path}")
            except Exception as e:
                self.result_display.setText(f"모델 불러오기 중 오류 발생:\n\n{e}")

# --- 4. 애플리케이션 실행 ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
