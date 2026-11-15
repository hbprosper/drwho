#-------------------------------------------------------------------------
# Real time monitoring of loss curves during training
# Harrison B. Prosper
# Created: July 2021
# Updated: Jun 12 2026 - Use tensorboard by default, if installed.
# Updated: Aug 21 2026 - Use standalone monitor by defualt!
# Updated: Sep 22 2026 - Import tensorboard lazily.
# Updated: Oct  4 2026 - Draw in the notebook when there is no display.
# Updated: Oct 19 2026 - Fix: record the best validation loss (min_avloss typo),
#                         so params are saved only when the loss improves.
# Updated: Nov 15 2026 - Remove the flicker of the in-notebook loss plot:
#                         update the output in place, fix the axes frame.
#-------------------------------------------------------------------------
import os, sys, re
import subprocess
import numpy as np
from time import sleep
from pathlib import Path
# module for shell utilities
import shutil
# tensorboard is optional and is imported lazily by _summary_writer() below:
# torch.utils.tensorboard is slow to import, and most users never ask for it.
#-------------------------------------------------------------------------
try:
    import pandas as pd
except:
    raise ImportError('''
    Please install pandas:

        conda install pandas
        
    ''')
import time
try:
    import matplotlib as mp
    import matplotlib.pyplot as plt
    # update fonts
    plt.rcParams.update({
  "text.usetex": shutil.which('latex') is not None,
  "font.family": "sans-serif",
  "font.size": 14
    })
except:
    raise ImportError('''
    Please install matplotlib:

        conda install matplotlib
    ''')
#-------------------------------------------------------------------------
DELAY = 5 # seconds - interval between plot updates
LOG_SWITCH = 3
CHECK = "\u2705"
FAIL  = "\u274C"
WARN  = "\u26A0"
FONTSIZE = 14
#-------------------------------------------------------------------------
# The loss file should be a simple text file with olumns of numbers:
#
#   iterations,train-losses,validation-losses,...
#     
def get_losses(loss_file):
    try:
        losses = pd.read_csv(loss_file).to_numpy()
        return losses[:, 0], losses[:, 1], losses[:, 2]
    except:
        return None

def get_timeleft(timeleft_file):
    if timeleft_file == None:
        return None
    try:
        return open(timeleft_file, 'r').read().strip()
    except:
        return None

def elapsed_time(now, start):
    etime = now() - start    
    t = etime
    hours = int(t / 3600)
    t = t - 3600 * hours
    minutes = int(t / 60)
    seconds = t - 60 * minutes
    etime_str = "%2.2d:%2.2d:%2.2d" % (hours, minutes, seconds)
    return etime_str, etime, (hours, minutes, seconds)

def plot_loss_curve(losses, filename='fig_losses.png'):
    
    xx, yy_t, yy_v = losses
    
    # create an empty figure
    fig = plt.figure(figsize=(4*16/9, 4))

    # add a subplot to it
    nrows, ncols, index = 1,1,1
    ax  = fig.add_subplot(nrows,ncols,index)
    
    ax.plot(xx, yy_t, color='red',  lw=1, linestyle='dashed', label='training loss')
    ax.plot(xx, yy_v, color='blue', lw=1, label='validation loss')
    ax.legend()
    
    ax.set_xlabel('iterations', fontsize=FONTSIZE)
    ax.set_ylabel('average loss', fontsize=FONTSIZE)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='-')
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_losses(loss_file, filename='fig_losses.png'):
    losses = get_losses(loss_file)
    if losses is not None:
        plot_loss_curve(losses, filename)
    else:
        print(f'''
{WARN} loss file {loss_file} not found!
        ''')
    
class TimeLeft:
    '''
    Return the amount of time left.
    
    timeleft = TimeLeft(N)
    
    N: maximum loop count
    
      for i in timeleft:
          : :

    or
       timeleft(i, extra)
      
    '''
    def __init__(self, N):
        self.N = N        
        self.timenow = time.time
        self.start = self.timenow()
        self.str = ''
        
    def __del__(self):
        pass
    
    def __timestr(self, ii):
        # elapsed time since start
        elapsed = self.timenow() - self.start
        s = elapsed
        h = int(s / 3600) 
        s = s - 3600*h
        m = int(s / 60)
        s = s - 60*m
        hh= h
        mm= m
        ss= s
        
        # time/loop
        count = ii+1
        t = elapsed / count
        f = 1/t if count > 10 else 0.0
        
        # time left
        s = t * (self.N - count)
        h = int(s / 3600) 
        s = s - 3600*h
        m = int(s / 60)
        s =  s - 60*m
        percent = 100 * count / self.N

        return "%10d|%6.2f%s|%2.2d:%2.2d:%2.2d|%2.2d:%2.2d:%2.2d|%6.1f it/s" % \
            (ii, percent, '%', hh, mm, ss, h, m, s, f)
        
    def __iter__(self):
        
        for ii in range(self.N):
            
            if ii < self.N-1:
                print(f'\r{self.__timestr(ii):s}', end='')
            else: 
                print(f'\r{self.__timestr(ii):s}')
                
            yield ii
            
    def __call__(self, ii, extra='', colorize=False):    
        if extra != '':
            if colorize:
               extra = "\x1b[1;34;48m|%s\x1b[0m" % extra
                
        self.a_str = f'{self.__timestr(ii):s}{extra:s}'
        return self.a_str

    def __str__(self):
        return self.a_str()
#--------------------------------------------------------------------
class LossMonitor:    
    '''    
    monitor = LossMonitor(lossfile, [ylabel=R(omega), ylog=True, xlog=False])
        :   :
    monitor()
    '''
    def __init__(self, lossfile, 
                 ylabel='$R(\\omega)$', 
                 ylog=True, 
                 xlog=False):

        self.lossfile = lossfile
        self.timeleftfile = lossfile.replace('.csv', '.txt')
        self.ylabel = ylabel
        self.ylog = ylog
        self.xlog = xlog
        
        # get first blocking backend
        self.original_backend = mp.get_backend()
        for backend in ("TkAgg", "QtAgg", "MacOSX"):
            try:
                mp.use(backend, force=True)
                break
            except:
                backend = None
        if backend == None:
            print(f'{WARN} No suitable GUI (blocking) backend found for Monitor!')
        else:
            print(f'\nplotting backend: {backend}')
        import matplotlib.pyplot as plt

        # set up an empty figure
        self.fig = plt.figure(figsize=(4*16/9, 4))
        self.fig.suptitle(self.lossfile)

        # add a subplot to it
        nrows, ncols, index = 1,1,1
        self.ax = self.fig.add_subplot(nrows, ncols, index)

    def __update(self, frame=None):            
        fig, ax = self.fig, self.ax
        
        ax.clear()
        ax.set_xlabel('Iteration', fontsize=14)
        ax.set_ylabel(self.ylabel, fontsize=14)
        ax.grid(True, which="both", linestyle='-')
        
        data = get_losses(self.lossfile)
        if type(data) != type(None):
            
            iters, train, valid = data
            
            if len(train) > 0:

                if self.ylog:
                    if train[0]/train[-1] > LOG_SWITCH:
                        ax.set_yscale('log')
                    
                if self.xlog:
                    if len(iters) > 10:
                        ax.set_xscale('log')
                    
                timeleft = get_timeleft(self.timeleftfile)
                if timeleft != None:
                    ax.set_title(timeleft, fontsize=11)
                else:
                    ax.set_title('Iteration: %5d|%s' % (iters[-1], time.ctime()))
                    
                ax.plot(iters, train, c='red',  linestyle='dashed', label='training')
                ax.plot(iters, valid, c='blue', label='validation')
                ax.legend()
                
        fig.tight_layout()
        
    def show(self):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        self.anim = FuncAnimation(
            fig=self.fig, 
            func=self.__update, 
            interval=1000*DELAY, # milliseconds
            repeat=False, 
            cache_frame_data=False)

        # this should block
        plt.show()
        print('\nciao!')
#--------------------------------------------------------------------
def _has_display():
    '''
    Return True if a GUI window can plausibly be opened on this machine.

    Google Colab and other headless sessions cannot pop up a window, so the
    Monitor must not launch the standalone monitor process there: it would
    die at once, and because its output is discarded the failure would be
    silent.
    '''
    if 'google.colab' in sys.modules:
        return False

    # macOS and Windows manage their own windows. On Linux an unset DISPLAY
    # means there is no X server to draw on.
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        return False

    return True

def _in_notebook():
    '''Return True if running inside IPython, Jupyter or Colab.'''
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False

def _summary_writer():
    '''
    Return tensorboard's SummaryWriter class, or None if tensorboard is not
    installed. The import happens here rather than at module level because
    torch.utils.tensorboard is slow to import and is needed only when a
    Monitor is created with use_tensorboard=True.
    '''
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    return SummaryWriter

def tensorboard_available():
    '''Return True if tensorboard can be imported.'''
    return _summary_writer() is not None
#--------------------------------------------------------------------
class Monitor:
    '''
    Write training and validation losses to a csv file and optionally
    the model parameters. By default, the losses are displayed by a 
    simple standalone display is activated in a separate window.
    However, if tensorboard is available, and desired, set
    use_tensorboard=True.
    '''

    def __init__(self, 
                 niterations, 
                 lossfile,  
                 monitorstep,
                 newlossfile=True,
                 frac=0.005,
                 model=None, 
                 paramsfile=None,
                 use_tensorboard=False,
                 ylabel=None):
      
        # cache inputs
        self.niterations = niterations
        self.lossfile    = lossfile
        self.monitorstep = monitorstep
        self.newlossfile = newlossfile
        self.frac  = frac
        self.model = model
        self.paramsfile  = paramsfile
        
        p = Path(lossfile)
        self.timeleftfile= lossfile.replace('.csv', '.txt')
        self.checkfile = lossfile.replace(f'{p.name}', 'checkpoint.csv')
        
        self.minavloss = float('inf')  # initialize minimum average loss
        self.ylabel = ylabel

        # get absolute path to runs folder
        # runs_pathname = p.parent.resolve()

        self.writer = None
        self.inline = False   # draw in the notebook instead of a window
        if use_tensorboard:
            # import tensorboard only now, when it has actually been asked for
            SummaryWriter = _summary_writer()
            if SummaryWriter is None:
                print(f'{WARN} tensorboard not installed, '
                      'falling back to the builtin loss monitor.')
            else:
                self.writer = SummaryWriter(
                    log_dir=os.path.expanduser('~/runs'))
            
        # In case the graphics backend changes, let's 
        # cache current backend and restore in the end function
        self.original_backend = mp.get_backend()
    
        # initialize loss file
        # create loss file if it does not exist
        if not os.path.exists(lossfile) or newlossfile:
            open(lossfile, 'w').write('iteration,train,val,valbest,lr\n')  
        
        self.reset()
        
    def __call__(self, t_loss, v_loss, lr=0, epoch=None, same_line=True):
        
        loss_decreased = v_loss < (1 - self.frac) * self.minavloss
        if loss_decreased:
            self.minavloss = v_loss   # remember the new best validation loss
        v_best_loss = self.minavloss
        
        # update loss file
        jj = self.ii-1 # the update occurs in  step()
        self.itno = self.offset + jj

        open(self.lossfile, 
             'a').write(f'{self.itno:10d},'
                        f'{t_loss:9.3e},{v_loss:9.3e},'
                        f'{v_best_loss:9.3e},{lr:9.3e}\n')

        if self.writer is not None:
            self.writer.add_scalar("Loss/train", t_loss, self.itno)
            self.writer.add_scalar("Loss/val",   v_loss, self.itno)
            self.writer.add_scalar("Loss/val_best", v_best_loss, self.itno)
            self.writer.add_scalar("LearningRate", lr, self.itno)
            self.writer.flush()
        
        # if specified save model parameters
        if type(self.model) != type(None):
            if loss_decreased:
                
                if os.path.exists(self.paramsfile):
                    cmd = f'mv {self.paramsfile} {self.paramsfile}.previous'
                    os.system(cmd)
                    
                self.model.save(self.paramsfile)

                if os.path.exists(self.checkfile):
                    cmd = f'mv {self.checkfile} {self.checkfile}.previous'
                    os.system(cmd)
                    
                open(self.checkfile, 'w').write(
                    'iteration,train,val,valbest,lr\n') 
                open(self.checkfile, 'a').write(
                    f'{self.itno:10d},'
                    f'{t_loss:9.3e},{v_loss:9.3e},'
                    f'{v_best_loss:9.3e},{lr:9.3e}\n')

        # update time left file
        if epoch != None:
            line = f'|{t_loss:9.3e}|{v_loss:9.3e}|{epoch:10d}|'
        else:
            line = f'|{t_loss:9.3e}|{v_loss:9.3e}|{self.itno:10d}|'
            
        timeleft_str = self.timeleft(jj, line)

        if self.inline:
            self._inline_update(timeleft_str)
        elif same_line:
            print(f'\r{timeleft_str}', end='')
        else:
            print(timeleft_str)
            
        open(self.timeleftfile, 'w').write(f'{timeleft_str:s}\n')
        
    def step(self):
        save = self.ii % self.monitorstep == 0
        self.ii += 1
        return save

    def read_checkpoint(self):
        if sys.path.exists(self.checkfile):
            return pd.read_csv(self.checkfile)
        else:
            return None
        
    def reset(self):

        if self.newlossfile:
            open(self.lossfile, 'w').write('iteration,train,val,valbest,lr\n')
            self.offset = 0
        else:
            # get last iteration number from loss file
            df = pd.read_csv(self.lossfile)
    
            # initialize iteration counters: absolute and relative
            if len(df) < 1:
                self.offset = 0
            else:
                self.offset = df.iteration.iloc[-1] # get last iteration number

        # relative counter
        self.ii = 0

        self.timeleft = TimeLeft(self.niterations)

    def start(self):

        self.reset()

        # Handles to the in-notebook outputs (loss plot and status line).
        # They are created by the first call to _inline_update() and then
        # updated in place. Reset them here so that a second training run
        # in the same notebook draws a new plot rather than updating the
        # plot of the previous run.
        self._plot_handle   = None
        self._status_handle = None

        if self.writer is not None:
            # tensorboard is handling the display
            return

        if _has_display():
            # a GUI window is possible: launch the standalone monitor
            cmd = ["monlosses", self.lossfile]
            if self.ylabel != None:
                cmd.append(self.ylabel)
            print(' '.join(cmd))

            self.p = subprocess.Popen(cmd,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL,
                                      start_new_session=True)

        elif _in_notebook():
            # Colab, or any headless notebook: no window can be opened, so
            # draw the loss curve in the output cell instead.
            self.inline = True
            print('\tNo display available; '
                  'drawing the loss curve in the notebook.')

        else:
            # headless and not a notebook: there is nothing to draw on
            print(f'{WARN} No display and not a notebook. Losses are being '
                  f'written to\n\t{self.lossfile}\n\tbut not displayed. '
                  'Run "monlosses <lossfile>" elsewhere to watch them.')

    def _inline_update(self, status=''):
        '''
        Redraw the loss curve inside the notebook output cell.

        Used when no GUI window can be opened (Google Colab, or any headless
        notebook), where the standalone monitor process cannot work.

        How flicker is avoided
        ----------------------
        1. The output is updated *in place*. The first call displays the
           plot (and the status line) with a display handle; later calls
           replace the contents of that handle. The earlier version cleared
           the output cell and displayed a new image each time, which made
           the cell collapse and re-expand, so that the page jumped.

        2. The axes frame is fixed. Log-log axes are used from the start;
           the x-axis spans the whole run, from the first iteration to the
           last, so the curve grows into a fixed frame; and the y-axis
           limits are whole powers of ten, so they change only when the
           loss crosses into a new decade. Previously, matplotlib chose the
           limits from the data at every update, so the curve shifted and
           rescaled each time.

        3. The margins are fixed. tight_layout() recomputes the margins from
           the widths of the tick labels, which change as the limits change,
           so the plotting area moved sideways. Fixed margins keep it still.
        '''
        try:
            from IPython.display import display, Image, Pretty
        except ImportError:
            if status:
                print(status)
            return

        # Build the figure with a bare Figure + Agg canvas rather than
        # pyplot: no global state, no backend switching, nothing to close.
        import io
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        # handles to the outputs (see start()); getattr guards against
        # _inline_update() being called without start() having been called
        plot_handle   = getattr(self, '_plot_handle', None)
        status_handle = getattr(self, '_status_handle', None)

        # On the first call, reserve the plot's place with an empty output,
        # before the status line is displayed. Otherwise, since the first
        # update usually has too few points to draw a curve, the status line
        # would be displayed first and the plot would end up below it.
        if plot_handle is None:
            plot_handle = self._plot_handle = display(Pretty(''),
                                                      display_id=True)

        losses = get_losses(self.lossfile)

        have_points = False

        if losses is not None:

            xx, yy_t, yy_v = losses

            # a log scale needs positive values: drop any non-positive
            # iteration numbers (e.g., iteration 0) and losses
            keep = (xx > 0) & (yy_t > 0) & (yy_v > 0)
            xx, yy_t, yy_v = xx[keep], yy_t[keep], yy_v[keep]

            # a curve needs at least two points
            have_points = len(xx) > 1

        if have_points:

            fig = Figure(figsize=(6, 6 * 9 / 16))

            # fixed margins (fractions of the figure width and height),
            # chosen to leave room for the axis labels
            fig.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.96)

            ax  = fig.add_subplot(1, 1, 1)

            ax.plot(xx, yy_t, color='red',  lw=1, label='training loss')
            ax.plot(xx, yy_v, color='blue', lw=1, label='validation loss')

            # log-log from the start (no switch from linear to log)
            ax.set_xscale('log')
            ax.set_yscale('log')

            # x: from the first iteration recorded to the last iteration
            # of this run. offset is nonzero when training resumes and
            # appends to an existing loss file (newlossfile=False).
            xmin = xx[0]
            xmax = max(self.offset + self.niterations, xx[-1])
            ax.set_xlim(xmin, xmax)

            # y: snap the limits outward to whole powers of ten
            y = np.concatenate([yy_t, yy_v])
            ymin = 10**np.floor(np.log10(y.min()))
            ymax = 10**np.ceil(np.log10(y.max()))
            if ymax <= ymin:
                ymax = 10 * ymin   # all losses are an exact power of ten
            ax.set_ylim(ymin, ymax)

            # fixed legend position: 'best' would move as the curve grows
            ax.legend(loc='upper right')
            ax.set_xlabel('iterations', fontsize=FONTSIZE)
            ax.set_ylabel('average loss' if self.ylabel is None else self.ylabel,
                          fontsize=FONTSIZE)

            ax.grid(True, which='both', linestyle='-')

            # Render the PNG here rather than handing IPython the Figure:
            # IPython only knows how to draw a Figure once pyplot has been
            # imported and registered a formatter for it, which a library
            # cannot assume has happened.
            buf = io.BytesIO()
            FigureCanvasAgg(fig).print_png(buf)
            image = Image(data=buf.getvalue())

            # replace the contents of the plot's output in place
            plot_handle.update(image)

        if status:
            # The status line also gets its own handle, so that it, too,
            # is updated in place below the plot, rather than appended.
            text = Pretty(status)

            if status_handle is None:
                self._status_handle = display(text, display_id=True)
            else:
                status_handle.update(text)

    def terminate(self, delay=5):

        if self.inline:
            # drawing in the notebook: one final redraw, nothing to tear down
            self._inline_update()
            print('\tDone!')
            return

        mp.use(self.original_backend, force=True)

        sleep(delay)

        if self.writer is None:

            if not hasattr(self, 'p'):
                # no standalone monitor was ever started (no display and no
                # notebook), so there is nothing to shut down
                return

            # kill standalone monitoring process
            try:
                print('\n\tTerminating standalone loss monitor...')
                self.p.terminate()
                try:
                    self.p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.p.kill()
                    self.p.wait()
                print('\tDone!')
            except:
                print('\tNone started in this session!')

        else:
            self.writer.flush()
            self.writer.close()
