#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gmxplot GUI - XVG plotter for GROMACS molecular dynamics data.
Enhanced with scientific styles, accessible palettes, statistics, and curve fitting.
Requires: matplotlib, numpy, pandas, scipy, seaborn
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns
import re

# Attempt to import scienceplots (fail silently if not available or incompatible)
try:
    import scienceplots
    HAS_SCIENCEPLOTS = True
except Exception:
    HAS_SCIENCEPLOTS = False
    # Fallback styles from matplotlib
    SCIENCE_STYLES = ["default", "seaborn-v0_8", "ggplot", "fivethirtyeight"]
else:
    SCIENCE_STYLES = ["default", "science", "ieee", "nature", "scatter"]

# =============================================================================
# XVG file parsing functions (adapted from original GMXvg)
# =============================================================================
def parse_xvg(filepath):
    """
    Read an .xvg file and return a dict with:
        - 'data': DataFrame with numerical data
        - 'metadata': dict with axis labels, legends, etc.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    metadata = {}
    data_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue
        elif line.startswith('@'):
            # Extract metadata like @ xaxis label "Time (ps)" or @ legend "text"
            if 'xaxis' in line or 'yaxis' in line or 'legend' in line:
                matches = re.findall(r'"([^"]*)"', line)
                if matches:
                    if 'xaxis' in line:
                        metadata['xlabel'] = matches[0]
                    elif 'yaxis' in line:
                        metadata['ylabel'] = matches[0]
                    elif 'legend' in line:
                        metadata['legend'] = matches
            continue
        else:
            # Data line: try to convert all tokens to float
            tokens = line.split()
            if not tokens:
                continue
            try:
                row = [float(x) for x in tokens]
                data_lines.append(row)
            except ValueError:
                # Skip line if any token is not numeric
                continue

    if not data_lines:
        return None

    # Convert to DataFrame
    df = pd.DataFrame(data_lines)

    # Assign column names from legends or generic
    if 'legend' in metadata:
        legends = metadata['legend']
        if len(legends) == df.shape[1]:
            df.columns = legends
        else:
            df.columns = [f'Col_{i}' for i in range(df.shape[1])]
    else:
        df.columns = [f'Col_{i}' for i in range(df.shape[1])]

    # Default metadata
    metadata.setdefault('xlabel', 'X')
    metadata.setdefault('ylabel', 'Y')
    metadata['title'] = os.path.basename(filepath)

    return {'data': df, 'metadata': metadata}

# =============================================================================
# Curve fitting functions
# =============================================================================
def linear_func(x, a, b):
    return a * x + b

def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

FIT_FUNCTIONS = {
    'Linear': linear_func,
    'Quadratic': quadratic_func,
    'Exponential': exponential_func,
}

# =============================================================================
# Main GUI class
# =============================================================================
class GMXvgGUI:
    def __init__(self, master):
        self.master = master
        master.title("gmxplot - XVG Plotter")
        master.geometry("1100x750")

        # State variables
        self.directory = tk.StringVar()
        self.file_list = []          # full paths
        self.current_data = None     # dict with data and metadata

        # Plot options
        self.plot_type = tk.StringVar(value="line")
        self.style_choice = tk.StringVar(value="default")
        self.palette_choice = tk.StringVar(value="colorblind")
        self.show_legend = tk.BooleanVar(value=True)
        self.show_stats = tk.BooleanVar(value=False)
        self.show_fit = tk.BooleanVar(value=False)
        self.fit_func = tk.StringVar(value="Linear")
        self.x_col = tk.StringVar(value="Col_0")

        # Build UI
        self.create_widgets()

    def create_widgets(self):
        # Left panel (controls)
        left_frame = ttk.Frame(self.master, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Row counters
        row = 0

        # Directory selection
        ttk.Label(left_frame, text="Working directory:").grid(row=row, column=0, sticky=tk.W)
        row += 1
        ttk.Entry(left_frame, textvariable=self.directory, width=40).grid(row=row, column=0, padx=5, pady=5)
        ttk.Button(left_frame, text="Browse...", command=self.browse_dir).grid(row=row, column=1, padx=5)
        row += 1
        ttk.Button(left_frame, text="Load .xvg files", command=self.load_files).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        # File list
        ttk.Label(left_frame, text="XVG files found:").grid(row=row, column=0, sticky=tk.W, pady=(10,0))
        row += 1
        self.file_listbox = tk.Listbox(left_frame, selectmode=tk.EXTENDED, height=10, width=50)
        self.file_listbox.grid(row=row, column=0, columnspan=2, pady=5)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        row += 1

        # Plot type and Generate button
        ttk.Label(left_frame, text="Plot type:").grid(row=row, column=0, sticky=tk.W, pady=(10,0))
        row += 1
        plot_types = [("Line", "line"), ("Bar", "bar"), ("Boxplot", "box"), ("Histogram", "hist")]
        for i, (text, val) in enumerate(plot_types):
            ttk.Radiobutton(left_frame, text=text, variable=self.plot_type, value=val).grid(row=row+i, column=0, sticky=tk.W)
        # Place the Generate button next to the last radio button (same row, column=1)
        ttk.Button(left_frame, text="Generate Plot", command=self.plot_selected).grid(row=row+len(plot_types)-1, column=1, padx=10, sticky=tk.W)
        row += len(plot_types)

        # Style (SciencePlots or alternatives)
        ttk.Label(left_frame, text="Style:").grid(row=row, column=0, sticky=tk.W, pady=(10,0))
        row += 1
        self.style_menu = ttk.Combobox(left_frame, textvariable=self.style_choice, values=SCIENCE_STYLES, state="readonly", width=20)
        self.style_menu.grid(row=row, column=0, columnspan=2, pady=5)
        self.style_menu.set("default")
        row += 1

        # Color palette
        ttk.Label(left_frame, text="Color palette:").grid(row=row, column=0, sticky=tk.W, pady=(10,0))
        row += 1
        palettes = ["colorblind", "viridis", "plasma", "Set1", "Set2", "Paired", "Dark2"]
        self.palette_menu = ttk.Combobox(left_frame, textvariable=self.palette_choice, values=palettes, state="readonly", width=20)
        self.palette_menu.grid(row=row, column=0, columnspan=2, pady=5)
        self.palette_menu.set("colorblind")
        row += 1

        # Checkboxes for features
        ttk.Checkbutton(left_frame, text="Show legend", variable=self.show_legend).grid(row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        ttk.Checkbutton(left_frame, text="Annotate mean and std", variable=self.show_stats).grid(row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        ttk.Checkbutton(left_frame, text="Curve fitting", variable=self.show_fit).grid(row=row, column=0, sticky=tk.W, pady=2)
        row += 1

        # Fit function selection
        ttk.Label(left_frame, text="Fit function:").grid(row=row, column=0, sticky=tk.W, pady=(10,0))
        row += 1
        fit_funcs = list(FIT_FUNCTIONS.keys())
        self.fit_menu = ttk.Combobox(left_frame, textvariable=self.fit_func, values=fit_funcs, state="readonly", width=20)
        self.fit_menu.grid(row=row, column=0, columnspan=2, pady=5)
        self.fit_menu.set("Linear")
        row += 1

        # Column selection
        ttk.Label(left_frame, text="X column:").grid(row=row, column=0, sticky=tk.W, pady=(10,0))
        row += 1
        self.x_col_menu = ttk.Combobox(left_frame, textvariable=self.x_col, state="readonly", width=20)
        self.x_col_menu.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        ttk.Label(left_frame, text="Y columns (Ctrl+click):").grid(row=row, column=0, sticky=tk.W, pady=(10,0))
        row += 1
        self.y_col_listbox = tk.Listbox(left_frame, selectmode=tk.EXTENDED, height=4, width=30)
        self.y_col_listbox.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        # Export buttons (at the bottom)
        ttk.Button(left_frame, text="Export Plot...", command=self.export_plot).grid(row=row, column=0, pady=5)
        ttk.Button(left_frame, text="Export Data (CSV)", command=self.export_csv).grid(row=row, column=1, pady=5)
        row += 1

        # Right panel (figure)
        right_frame = ttk.Frame(self.master, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Matplotlib figure
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar_frame = ttk.Frame(right_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    # =========================================================================
    # GUI methods
    # =========================================================================
    def browse_dir(self):
        dir_selected = filedialog.askdirectory()
        if dir_selected:
            self.directory.set(dir_selected)
            self.load_files()

    def load_files(self):
        """Search for .xvg files in the directory and list them."""
        path = self.directory.get()
        if not path or not os.path.isdir(path):
            messagebox.showerror("Error", "Invalid directory.")
            return
        self.file_list = []
        self.file_listbox.delete(0, tk.END)
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.xvg'):
                    full = os.path.join(root, f)
                    self.file_list.append(full)
                    self.file_listbox.insert(tk.END, f)
        if not self.file_list:
            messagebox.showinfo("Info", "No .xvg files found.")

    def on_file_select(self, event):
        """When a file is selected, load it and update column menus, then auto-plot."""
        selection = self.file_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        filepath = self.file_list[idx]
        try:
            parsed = parse_xvg(filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{e}")
            return
        if parsed is None:
            messagebox.showerror("Error", f"No numerical data found in {filepath}.")
            return
        self.current_data = parsed
        df = parsed['data']
        columns = df.columns.tolist()
        # Update column menus
        self.x_col_menu['values'] = columns
        self.x_col_menu.set(columns[0])
        self.y_col_listbox.delete(0, tk.END)
        for col in columns[1:]:
            self.y_col_listbox.insert(tk.END, col)
        # Select all Y columns by default
        for i in range(self.y_col_listbox.size()):
            self.y_col_listbox.selection_set(i)

        # Auto-generate the plot
        self.plot_selected()

    def get_selected_y_cols(self):
        """Return list of selected Y column names."""
        selected_indices = self.y_col_listbox.curselection()
        if not selected_indices:
            selected_indices = range(self.y_col_listbox.size())
        return [self.y_col_listbox.get(i) for i in selected_indices]

    def plot_selected(self):
        """Generate the plot according to selected options."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please select an .xvg file first.")
            return

        try:
            df = self.current_data['data']
            metadata = self.current_data['metadata']
            x_col = self.x_col.get()
            y_cols = self.get_selected_y_cols()
            if not y_cols:
                messagebox.showerror("Error", "Select at least one Y column.")
                return

            x_data = df[x_col].values
            y_data_list = [df[col].values for col in y_cols]

            # Apply style
            self.apply_style()

            # Clear axes
            self.ax.clear()

            plot_type = self.plot_type.get()
            palette = self.palette_choice.get()
            colors = sns.color_palette(palette, n_colors=len(y_cols))

            # Plot according to type
            if plot_type == "line":
                for i, y_data in enumerate(y_data_list):
                    self.ax.plot(x_data, y_data, label=y_cols[i], color=colors[i])
            elif plot_type == "bar":
                if len(y_cols) > 1:
                    # Grouped bars
                    bar_width = 0.8 / len(y_cols)
                    x_pos = np.arange(len(x_data))
                    for i, y_data in enumerate(y_data_list):
                        offset = (i - (len(y_cols)-1)/2) * bar_width
                        self.ax.bar(x_pos + offset, y_data, width=bar_width, label=y_cols[i], color=colors[i])
                    self.ax.set_xticks(x_pos)
                    self.ax.set_xticklabels([f'{v:.2f}' for v in x_data], rotation=45)
                else:
                    # Single bar with standard error
                    y_data = y_data_list[0]
                    error = np.std(y_data) / np.sqrt(len(y_data))
                    self.ax.bar(x_data, y_data, yerr=error, label=y_cols[0], color=colors[0])
            elif plot_type == "box":
                data_to_plot = [df[col].values for col in y_cols]
                self.ax.boxplot(data_to_plot, labels=y_cols, patch_artist=True,
                                boxprops=dict(facecolor='lightblue'))
            elif plot_type == "hist":
                y_data = y_data_list[0]
                self.ax.hist(y_data, bins=20, alpha=0.7, color=colors[0], label=y_cols[0])

            # Legend
            if self.show_legend.get() and plot_type in ["line", "bar"]:
                self.ax.legend()

            # Statistics annotation
            if self.show_stats.get():
                for i, y_data in enumerate(y_data_list):
                    mean_val = np.mean(y_data)
                    std_val = np.std(y_data)
                    x_pos = 0.95 * (max(x_data) - min(x_data)) + min(x_data)
                    y_pos = 0.95 * (max(y_data) - min(y_data)) + min(y_data)
                    self.ax.annotate(f'{y_cols[i]}: μ={mean_val:.3f} σ={std_val:.3f}',
                                     xy=(x_pos, y_pos - i*0.1*max(y_data)),
                                     xycoords='data', fontsize=8, color=colors[i])

            # Curve fitting (only for line plots)
            if self.show_fit.get() and plot_type == "line":
                fit_func_name = self.fit_func.get()
                fit_func = FIT_FUNCTIONS[fit_func_name]
                y_data = y_data_list[0]
                mask = ~np.isnan(y_data) & ~np.isnan(x_data)
                x_clean = x_data[mask]
                y_clean = y_data[mask]
                if len(x_clean) < 3:
                    messagebox.showwarning("Warning", "Not enough data points for fitting.")
                else:
                    try:
                        if fit_func_name == 'Exponential':
                            p0 = [1, 0.1, 0]
                        else:
                            p0 = [1, 1]
                        popt, pcov = curve_fit(fit_func, x_clean, y_clean, p0=p0, maxfev=5000)
                        x_fit = np.linspace(min(x_clean), max(x_clean), 200)
                        y_fit = fit_func(x_fit, *popt)
                        self.ax.plot(x_fit, y_fit, '--', label=f'Fit {fit_func_name}', color='red')
                        eq_text = self.get_equation_text(fit_func_name, popt)
                        self.ax.text(0.05, 0.95, eq_text, transform=self.ax.transAxes,
                                     fontsize=9, verticalalignment='top',
                                     bbox=dict(facecolor='white', alpha=0.8))
                    except Exception as e:
                        messagebox.showerror("Fitting error", f"Could not fit: {e}")

            # Labels and title
            self.ax.set_xlabel(metadata.get('xlabel', 'X'))
            self.ax.set_ylabel(metadata.get('ylabel', 'Y'))
            self.ax.set_title(metadata.get('title', 'Plot'))
            self.ax.grid(True, linestyle='--', alpha=0.6)

            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Plot error", f"An error occurred:\n{e}")

    def apply_style(self):
        """Apply the selected style (SciencePlots or fallback)."""
        style = self.style_choice.get()
        if style == "default" or not HAS_SCIENCEPLOTS:
            try:
                plt.style.use('seaborn-v0_8')
            except:
                plt.style.use('default')
        else:
            try:
                plt.style.use([style, 'grid'])
            except Exception:
                plt.style.use('default')
        # Override some parameters for quality
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.dpi'] = 100

    def get_equation_text(self, func_name, params):
        """Generate equation text for display."""
        if func_name == 'Linear':
            return f'y = {params[0]:.3f}x + {params[1]:.3f}'
        elif func_name == 'Quadratic':
            return f'y = {params[0]:.3f}x² + {params[1]:.3f}x + {params[2]:.3f}'
        elif func_name == 'Exponential':
            return f'y = {params[0]:.3f}·exp({params[1]:.3f}x) + {params[2]:.3f}'
        return ''

    def export_plot(self):
        """Export current figure to file (PNG, PDF, SVG)."""
        if self.fig is None:
            return
        filetypes = [("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=filetypes)
        if filename:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Plot saved to {filename}")

    def export_csv(self):
        """Export current data to CSV."""
        if self.current_data is None:
            return
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if filename:
            self.current_data['data'].to_csv(filename, index=False)
            messagebox.showinfo("Success", f"Data saved to {filename}")

# =============================================================================
# Launch the application
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = GMXvgGUI(root)
    root.mainloop()
