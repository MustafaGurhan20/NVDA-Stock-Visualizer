"""
PatternRecognition_runnable.py

Auto-generated wrapper that converts and executes the uploaded Jupyter notebook
`/mnt/data/PatternRecognition.ipynb` into a runnable, self-contained Python script.

Goals:
- Execute each code cell in order while handling common notebook-only constructs (% magics, ! commands).
- Provide safe mocks for commonly-missing modules (yfinance, mplfinance, talib-ish, etc.) so the script
  can run in offline environments and still produce synthetic data when required.
- Save an executed notebook and a converted .py file under /mnt/data for inspection.

Usage:
    python PatternRecognition_runnable.py

Outputs (saved to /mnt/data):
- PatternRecognition_executed.ipynb  : the notebook with outputs filled where possible
- PatternRecognition_converted_auto.py: the pure-Python conversion of the notebook

Note: This wrapper tries to be conservative (it will skip bash magics and non-portable cells),
and it will substitute synthetic data where real data files are missing.
"""
import nbformat
import io, sys, os, traceback, types, json
from nbformat import v4 as nbfv4

NB_PATH = "/mnt/data/PatternRecognition.ipynb"
EXECUTED_OUT = "/mnt/data/PatternRecognition_executed.ipynb"
CONVERTED_OUT = "/mnt/data/PatternRecognition_converted_auto.py"

# --- Utilities: safe printing and simple mocks ---
def safe_print(*args, **kwargs):
    print(*args, **kwargs)

# Minimal mock factory for unavailable modules
class SimpleMockModule(types.ModuleType):
    def __init__(self, name, attrs=None):
        super().__init__(name)
        attrs = attrs or {}
        for k, v in attrs.items():
            setattr(self, k, v)
    def __getattr__(self, item):
        # return a function that warns when called
        def _warn(*a, **k):
            print(f"[mock:{self.__name__}] called attribute '{item}' with args={a} kwargs={k}")
            return None
        return _warn

# Provide a robust mock for yfinance used in many notebooks
import pandas as _pd
import numpy as _np
class MockTicker:
    def __init__(self, symbol):
        self.symbol = symbol
    def history(self, period='1mo', start=None, end=None, interval='1d', **kwargs):
        # create a synthetic time series
        try:
            if start is not None and end is not None:
                dates = _pd.date_range(start=start, end=end, freq='D')
            else:
                # default 90 days
                dates = _pd.date_range(end=_pd.Timestamp.today(), periods=90, freq='D')
            n = len(dates)
            base = 100 + _np.cumsum(_np.random.randn(n))
            df = _pd.DataFrame({
                'Open': base + _np.random.randn(n),
                'High': base + _np.abs(_np.random.randn(n)),
                'Low': base - _np.abs(_np.random.randn(n)),
                'Close': base + _np.random.randn(n),
                'Adj Close': base + _np.random.randn(n),
                'Volume': _np.random.randint(1000,10000,size=n)
            }, index=dates)
            return df
        except Exception as e:
            print("[mock yfinance] history failed:", e)
            return _pd.DataFrame()

def mock_download(tickers, start=None, end=None, period=None, interval='1d', **kwargs):
    if isinstance(tickers, str):
        return MockTicker(tickers).history(start=start, end=end, period=period, interval=interval, **kwargs)
    elif isinstance(tickers, (list, tuple)):
        return {t: MockTicker(t).history(start=start, end=end, period=period, interval=interval, **kwargs) for t in tickers}
    else:
        return {}

# install mocks into sys.modules for common packages that may be missing in offline envs
def install_offline_mocks():
    mocks = {}
    # yfinance
    yf = types.ModuleType('yfinance')
    yf.Ticker = MockTicker
    yf.download = mock_download
    mocks['yfinance'] = yf
    # mplfinance (simple stub)
    mpf = types.ModuleType('mplfinance')
    def _mpf_plot(*args, **kwargs):
        print('[mock:mplfinance] plot called with args', args, 'kwargs', kwargs)
    mpf.plot = _mpf_plot
    mocks['mplfinance'] = mpf
    # talib-like simple helpers
    talib = types.ModuleType('talib')
    talib.SMA = lambda arr, timeperiod=30: _pd.Series(arr).rolling(window=timeperiod, min_periods=1).mean().values
    mocks['talib'] = talib
    # seaborn stub
    sns = types.ModuleType('seaborn')
    sns.lineplot = lambda *a, **k: print('[mock:seaborn] lineplot called')
    mocks['seaborn'] = sns
    # insert into sys.modules
    for name, mod in mocks.items():
        if name not in sys.modules:
            sys.modules[name] = mod

# Preprocessor for code: remove cell magics and leading ! commands
def preprocess_code(src):
    out_lines = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('%%'):
            out_lines.append('# [CELL MAGIC SKIPPED] ' + line)
        elif stripped.startswith('%'):
            out_lines.append('# [MAGIC SKIPPED] ' + line)
        elif stripped.startswith('!'):
            out_lines.append('# [SHELL SKIPPED] ' + line)
        else:
            out_lines.append(line)
    return '\n'.join(out_lines)

# Safe exec of a code cell: capture stdout/stderr and attach outputs to the cell
import contextlib
@contextlib.contextmanager
def capture_stdout_stderr():
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield new_out, new_err
    finally:
        sys.stdout, sys.stderr = old_out, old_err

# Run the notebook
def run_notebook(nb_path="PatternRecognition.ipynb", executed_out="PatternRecognition_executed.ipynb", converted_out="PatternRecognition_converted_auto.py"):
    if not os.path.exists(nb_path):
        raise FileNotFoundError(f"Notebook not found at {nb_path}")
    install_offline_mocks()

    nb = nbformat.read(nb_path, as_version=4)
    exec_ns = {
        '__name__': '__main__',
        '__file__': converted_out,
        'input': lambda prompt='': ''  # avoid blocking input()
    }

    executed_nb = nbformat.v4.new_notebook()
    executed_nb.metadata = nb.metadata
    executed_cells = []

    # copy markdown cells and process code cells
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            executed_cells.append(nbfv4.new_markdown_cell(cell.source))
        elif cell.cell_type == 'code':
            src = preprocess_code(cell.source)
            out_cell = nbfv4.new_code_cell(cell.source)
            # Try executing
            with capture_stdout_stderr() as (out_buf, err_buf):
                try:
                    compiled = compile(src, '<notebook>', 'exec')
                    exec(compiled, exec_ns)
                    stdout_val = out_buf.getvalue()
                    stderr_val = err_buf.getvalue()
                    outputs = []
                    if stdout_val:
                        outputs.append(nbfv4.new_output('stream', name='stdout', text=stdout_val))
                    if stderr_val:
                        outputs.append(nbfv4.new_output('stream', name='stderr', text=stderr_val))
                    # No execute_result capture here (simple)
                    out_cell.outputs = outputs
                    out_cell.execution_count = None
                except Exception as e:
                    tb = traceback.format_exc()
                    out_cell.outputs = [nbfv4.new_output('error', ename=type(e).__name__, evalue=str(e), traceback=tb.splitlines())]
                    out_cell.execution_count = None
            executed_cells.append(out_cell)
        else:
            # raw or other
            executed_cells.append(nbfv4.new_raw_cell(cell.source if hasattr(cell, 'source') else ''))

    executed_nb.cells = executed_cells

    # Save executed notebook
    try:
        with open(executed_out, 'w', encoding='utf-8') as f:
            nbformat.write(executed_nb, f)
        safe_print(f"Executed notebook saved to: {executed_out}")
    except Exception as e:
        safe_print("Failed to save executed notebook:", e)

    # Export to a pure .py file by concatenating code cells
    try:
        with open(converted_out, 'w', encoding='utf-8') as f:
            f.write('# Auto-converted from ' + os.path.basename(nb_path) + '\n')
            f.write('# This file was generated to be runnable in an offline environment with mocks.\n\n')
            for cell in nb.cells:
                if cell.cell_type == 'markdown':
                    f.write('# ' + '-'*40 + '\n')
                    for line in cell.source.splitlines():
                        f.write('# ' + line + '\n')
                    f.write('\n')
                elif cell.cell_type == 'code':
                    f.write('# ' + '-'*40 + '\n')
                    f.write(preprocess_code(cell.source) + '\n\n')
        safe_print(f"Converted .py saved to: {converted_out}")
    except Exception as e:
        safe_print("Failed to save converted script:", e)

    return executed_out, converted_out

# If module executed directly, run the conversion + execution
if __name__ == '__main__':
    try:
        exec_nb, conv = run_notebook()
        safe_print('\nAll done. Files created (if notebook existed):')
        safe_print(' - ', exec_nb)
        safe_print(' - ', conv)
    except Exception as main_e:
        safe_print('Error running wrapper:', main_e)
        traceback.print_exc()
