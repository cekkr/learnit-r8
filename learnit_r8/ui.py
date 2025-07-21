import curses
import time

class CursesUI:
    """Manages the real-time terminal UI for training progress."""

    def __init__(self):
        try:
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            curses.start_color()
            curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
            self.height, self.width = self.stdscr.getmaxyx()
            self.start_time = time.time()
        except curses.error:
            self.stdscr = None
            print("Curses initialization failed. Running in text-only mode.")

    def _draw_progress_bar(self, y, x, width, progress, label):
        if self.stdscr is None: return
        
        bar_width = width - len(label) - 7
        if bar_width <= 0: return

        filled_len = int(bar_width * progress)
        bar = '█' * filled_len + '-' * (bar_width - filled_len)
        self.stdscr.addstr(y, x, f"{label} [{bar}] {progress*100:5.1f}%")

    def update(self, epoch, total_epochs, step, total_steps_in_epoch, global_step, total_global_steps, loss, lr):
        if self.stdscr is None:
            # Fallback for when curses is not available
            if step % 20 == 0: # Print every 20 steps to avoid spam
                 print(f"Epoch {epoch}/{total_epochs} | Step {step}/{total_steps_in_epoch} | Loss: {loss:.4f} | LR: {lr:.6f}")
            return

        self.stdscr.clear()
        
        # Title
        title = "learnit::r8 Training Dashboard"
        self.stdscr.addstr(0, (self.width - len(title)) // 2, title, curses.A_BOLD | curses.color_pair(1))

        # Overall Progress
        overall_progress = global_step / total_global_steps if total_global_steps > 0 else 0
        self._draw_progress_bar(2, 2, self.width - 4, overall_progress, "Total Progress:")
        
        # Epoch Progress
        epoch_progress = step / total_steps_in_epoch if total_steps_in_epoch > 0 else 0
        self._draw_progress_bar(3, 2, self.width - 4, epoch_progress, "Epoch Progress:")

        # Stats
        elapsed_time = time.time() - self.start_time
        eta = (elapsed_time / global_step) * (total_global_steps - global_step) if global_step > 0 else 0

        self.stdscr.addstr(5, 2, f"Epoch: {epoch}/{total_epochs}", curses.color_pair(2))
        self.stdscr.addstr(5, 20, f"Step: {step}/{total_steps_in_epoch}", curses.color_pair(2))
        self.stdscr.addstr(6, 2, f"Current Loss: {loss:.5f}", curses.color_pair(3))
        self.stdscr.addstr(6, 24, f"Learning Rate: {lr:.7f}", curses.color_pair(3))
        
        self.stdscr.addstr(8, 2, f"Time Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}", curses.A_DIM)
        self.stdscr.addstr(8, 26, f"Est. Time Remaining: {time.strftime('%H:%M:%S', time.gmtime(eta))}", curses.A_DIM)

        self.stdscr.refresh()

    def log_message(self, message, color_pair=4):
        if self.stdscr is None:
            print(message)
            return
        
        self.stdscr.addstr(self.height - 2, 2, " " * (self.width - 4)) # Clear line
        self.stdscr.addstr(self.height - 2, 2, message, curses.color_pair(color_pair))
        self.stdscr.refresh()

    def close(self):
        if self.stdscr:
            curses.nocbreak()
            self.stdscr.keypad(False)
            curses.echo()
            curses.endwin()