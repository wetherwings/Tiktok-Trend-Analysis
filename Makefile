#  Usage:
#    make install   → install all Python dependencies
#    make run       → execute baseline notebook end-to-end
#    make test      → run unit tests
#    make all       → install + run + test
#    make clean     → remove generated outputs
#
#  NOTE: transformer.ipynb requires a GPU — run it on
#        Google Colab (Runtime → T4 GPU → Run All)

PYTHON    := python3
PIP       := $(PYTHON) -m pip
VIZ_DIR   := visualizations
TEST_DIR  := test.py

.PHONY: all install run test clean help

# ── Default ───────────────────────────────────────────────────
all: install run test

# ── Install ───────────────────────────────────────────────────
install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -r requirements.txt --quiet
	@echo "✅ Dependencies installed"

# ── Run baseline (CPU, ~5 min) ────────────────────────────────
run:
	@echo "Running baseline notebook..."
	@echo "Datasets already in data/ folder (committed to repo)"
	@mkdir -p $(VIZ_DIR)
	jupyter nbconvert \
		--to notebook \
		--execute baseline.ipynb \
		--ExecutePreprocessor.timeout=600 \
		--output baseline_executed.ipynb
	@echo " Baseline complete → baseline_executed.ipynb"
	@echo ""
	@echo "   To run the transformer model:"
	@echo "   Open transformer.ipynb in Google Colab with T4 GPU"
	@echo "   Then run transformer_visualizations.ipynb for result figures"

# ── Tests ─────────────────────────────────────────────────────
test:
	@echo "Running unit tests..."
	$(PYTHON) -m pytest test.py -v --tb=short
	@echo " All tests passed"

# ── Clean ─────────────────────────────────────────────────────
clean:
	@echo "Cleaning generated files..."
	rm -f baseline_executed.ipynb
	rm -rf $(VIZ_DIR)/*.png $(VIZ_DIR)/*.html
	rm -rf __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete
	@echo " Cleaned"

# ── Help ──────────────────────────────────────────────────────
help:
	@echo ""
	@echo "TikTok Virality Prediction — CS 506"
	@echo "-------------------------------------"
	@echo "  make install  → install all dependencies"
	@echo "  make run      → run baseline notebook (CPU)"
	@echo "  make test     → run unit tests"
	@echo "  make all      → install + run + test"
	@echo "  make clean    → remove generated outputs"
	@echo ""
	@echo "  Transformer: open transformer.ipynb in Google Colab (T4 GPU)"
	@echo ""
