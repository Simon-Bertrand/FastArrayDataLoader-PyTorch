import pytest
import os
import shutil

@pytest.fixture
def temp_data_dir():
    path = "tests/data_functional"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    yield path
    if os.path.exists(path):
        try: shutil.rmtree(path)
        except: pass

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Hook to print benchmark results at the end of the session,
    visible even without -s.
    """
    file_path = "benchmark_results.md"
    if os.path.exists(file_path):
        terminalreporter.section("Benchmark Results (FastArrayDataLoader)")
        
        # Read the last few lines (e.g., current session) or the whole file?
        # Let's read the whole file to show context, or maybe just the last entries.
        # Since the file grows, let's just show the header and the last 5 lines.
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        if lines:
            # Print Header
            terminalreporter.write_line(lines[0].strip()) # Title
            terminalreporter.write_line(lines[2].strip()) # Table Header
            terminalreporter.write_line(lines[3].strip()) # Separator
            
            # Print last 5 entries
            last_entries = lines[4:]
            num_show = min(len(last_entries), 5)
            if num_show > 0:
                for line in last_entries[-num_show:]:
                    terminalreporter.write_line(line.strip())
            else:
                terminalreporter.write_line("(No results yet)")
        
        terminalreporter.write_line("")
