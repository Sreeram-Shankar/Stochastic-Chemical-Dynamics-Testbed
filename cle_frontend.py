import customtkinter as ctk
import subprocess, threading, os, zipfile, sys, subprocess
import numpy as np
from PIL import Image
from customtkinter import CTkImage
from tkinter import filedialog
import pandas as pd
from cle_backend import run_cle_simulation, run_ssa_simulation
import cle_visuals
import multiprocessing
import mpmath as mp
mp.dps = 200

#class that tracks simulation progress
class ProgressTracker:
    def __init__(self):
        self.current_step = 0
        self.total_steps = 1

#helper function for multiprocessing (must be at module level)
def create_plot_multiprocess(func_args):
    """Helper function for multiprocessing plot creation"""
    func, args = func_args
    return func(*args)

#sets the appearance of the oveall window
ctk.set_default_color_theme("theme.json")
ctk.set_appearance_mode("system")

#class that contains the window and all widgets
class CLEApp(ctk.CTk):
    #creates and configures the root
    def __init__(self):
        super().__init__()
        self.title("Chemical Langevin Equation Simulator")
        self.geometry("1100x800")
        self.resizable(False, False)
        self.build_gui()

    #function that builds all the gui components of the window
    def build_gui(self):
        #configures the grid layout of the window
        for i in range(50): self.grid_rowconfigure(i, weight=1)
        for j in range(4): self.grid_columnconfigure(j, weight=1)
        
        #initializes progress trackers (will be created when simulation starts)
        self.cle_progress_tracker = None
        self.ssa_progress_tracker = None
        self.cle_progress_bar = None
        self.cle_progress_label = None
        self.ssa_progress_bar = None
        self.ssa_progress_label = None
        self.simulation_running = False

        #defines the inputs and the kind of widget needed
        inputs = [
            ("Reaction Rate Constant k1:", "entry"), ("Reaction Rate Constant k2:", "entry"), 
            ("Reaction Rate Constant k3:", "entry"), ("Reaction Rate Constant k4:", "entry"),
            ("Reservoir Concentration A:", "entry"), ("Reservoir Concentration B:", "entry"),
            ("CLE Initial:", "entry"), ("SSA Initial:", "entry"),
            ("Final Time T:", "entry"), ("Number of Timesteps N:", "entry"), 
            ("Monte Carlo Paths M:", "entry"), ("Global Random Seed:", "entry"),
            ("Deterministic Integrator Family:", "dropdown", ["Explicit Runge-Kutta", "Adams-Bashforth", "Adams-Moulton", "SDIRK", "BDF", "Gauss-Legendre", "RadauIIA", "LobattoIIIC"]),
            ("Deterministic Integrator Order:", "entry"),
            ("Stochastic Integrator:", "dropdown", ["Euler–Maruyama", "Milstein", "Tamed Euler", "Balanced / Split-Step Euler"]),
            ("Operator Splitting:", "dropdown", ["Lie", "Strang"])
        ]
        
        #creates and places the main label
        self.main_label = ctk.CTkLabel(self, text="Chemical Langevin Equation Simulation - Please Enter Conditions", font=("Times New Roman", 34))
        self.main_label.grid(row = 0, rowspan=10, column=0, columnspan=4)

        #creates a dictionary of inputs to access
        self.widgets = {}

        #places the labels and widgets on the grid
        for i, (label_text, widget_type, *options) in enumerate(inputs):
            #defines the correct row and column
            col_offset = 0 if i < 8 else 2
            row_base = 10 + (i % 8) * 5

            #creates and places the column
            label = ctk.CTkLabel(self, text=label_text, font=("Times New Roman", 21))
            label.grid(row=row_base, column=col_offset, padx=10, pady=3, sticky="e")

            #configures and places the widget according to the type
            if widget_type == "entry":
                widget = ctk.CTkEntry(self, justify="left", placeholder_text="Enter...", font=("Times New Roman", 21))
            elif widget_type == "dropdown":
                values = options[0]
                widget = ctk.CTkOptionMenu(self, values=values, anchor="w", font=("Times New Roman", 21))
                widget.set(values[0])
            widget.grid(row=row_base, column=col_offset+1, padx=10, pady=5, sticky="w")
            self.widgets[label_text] = widget

        #creates the button to begin the calculations
        self.run_button = ctk.CTkButton(self, text="Begin", command=self.run_simulation, font=("Times New Roman", 25))
        self.run_button.grid(row=49, column=2, columnspan=2, padx=170, pady=6, sticky="nsew")

        #creates the button to toggle the theme of the window
        self.theme_toggle = ctk.CTkButton(self, text="Theme", command=self.toggle_theme, font=("Times New Roman", 25))
        self.theme_toggle.grid(row=49, column=0, columnspan=2, padx=100 ,pady=6, sticky="nsew")

    #function that begins the calculations
    def run_simulation(self):
        try:
            #collects the inputs from the user and makes sure that they are all valid numbers
            k1 = float(self.widgets["Reaction Rate Constant k1:"].get())
            k2 = float(self.widgets["Reaction Rate Constant k2:"].get())
            k3 = float(self.widgets["Reaction Rate Constant k3:"].get())
            k4 = float(self.widgets["Reaction Rate Constant k4:"].get())
            A = float(self.widgets["Reservoir Concentration A:"].get())
            B = float(self.widgets["Reservoir Concentration B:"].get())
            cle_initial = float(self.widgets["CLE Initial:"].get())
            ssa_initial = int(self.widgets["SSA Initial:"].get())
            T = float(self.widgets["Final Time T:"].get())
            N = int(self.widgets["Number of Timesteps N:"].get())
            M = int(self.widgets["Monte Carlo Paths M:"].get())
            random_seed = int(self.widgets["Global Random Seed:"].get())
            det_family = self.widgets["Deterministic Integrator Family:"].get()
            det_order = int(self.widgets["Deterministic Integrator Order:"].get())
            stoch_integrator = self.widgets["Stochastic Integrator:"].get()
            op_splitting = self.widgets["Operator Splitting:"].get()

            #checks for any logical errors in the code
            if(k1 <= 0 or k2 <= 0 or k3 <= 0 or k4 <= 0): 
                self.main_label.configure(text="Reaction Rate Constants Must be Positive")
                return

            if(A < 0 or B < 0): 
                self.main_label.configure(text="Reservoir Concentrations Must be Non-Negative")
                return

            if(cle_initial < 0): 
                self.main_label.configure(text="CLE Initial Must be Non-Negative")
                return

            if(ssa_initial < 0): 
                self.main_label.configure(text="SSA Initial Must be Non-Negative")
                return

            if(T <= 0): 
                self.main_label.configure(text="Final Time Must be Positive")
                return

            if(N < 1): 
                self.main_label.configure(text="Number of Timesteps Must be at Least 1")
                return

            if(M < 1): 
                self.main_label.configure(text="Monte Carlo Paths Must be at Least 1")
                return

            #checks if the determinsitic integrator order and family are valid
            if det_order < 1:
                self.main_label.configure(text="Deterministic Integrator Order Must be Positive")
                return

            if det_family.lower() == "explicit runge-kutta":
                if det_order < 1 or det_order > 7:
                    self.main_label.configure(text="Explicit Runge-Kutta Order Must be Between 1 and 7")
                    return
            elif det_family == "SDIRK":
                if det_order < 2 or det_order > 4:
                    self.main_label.configure(text="SDIRK Order Must be Between 2 and 4")
                    return
            elif det_family == "LobattoIIIC":
                if det_order <= 1:
                    self.main_label.configure(text="LobattoIIIC Order Must be Greater Than 1")
                    return

            #saves the results as class variables
            self.k1 = k1
            self.k2 = k2
            self.k3 = k3
            self.k4 = k4
            self.A = A
            self.B = B
            self.cle_initial = cle_initial
            self.ssa_initial = ssa_initial
            self.T = T
            self.N = N
            self.M = M
            self.random_seed = random_seed
            self.det_family = det_family
            self.det_order = det_order
            self.stoch_integrator = stoch_integrator
            self.op_splitting = op_splitting

            #calculates values based on inputs
            self.dt = T / N
            
        #displays an error message if type is invalid
        except Exception as e:
            self.main_label.configure(text="Please Make Sure Inputs are Valid Numbers of Correct Type")
            print(e)
            return
        
        #clears the results directory
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        for file in os.listdir(results_dir):
            file_path = os.path.join(results_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        #clears the graphs directory
        results_dir = "graphs"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        for file in os.listdir(results_dir):
            file_path = os.path.join(results_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        #removes all widgets from the screen
        for widget in self.winfo_children():
            if widget != self.main_label:
                widget.destroy()

        #configures the main label
        self.main_label.configure(text="Running CLE Simulation", font=("Times New Roman", 54))
        self.main_label.grid(row=0, rowspan=10, column=0, columnspan=4, pady=10, sticky="ew")
        
        #creates progress tracker for CLE
        self.cle_progress_tracker = ProgressTracker()
        
        #creates CLE progress bar and label in the middle of the screen
        self.cle_progress_label = ctk.CTkLabel(self, text="CLE Progress: 0/0", font=("Times New Roman", 28))
        self.cle_progress_label.grid(row=20, column=0, columnspan=4, pady=5, sticky="ew")
        
        self.cle_progress_bar = ctk.CTkProgressBar(self, orientation="horizontal", width=500, height=20, corner_radius=10, fg_color=("#ffffff", "#0d0d0d"), progress_color=("#ff8c42", "#ff8c42"))
        self.cle_progress_bar.set(0)
        self.cle_progress_bar.grid(row=21, column=0, columnspan=4, pady=5, sticky="ew")
        
        #SSA progress bar will be created later, after CLE is done
        self.ssa_progress_tracker = None
        self.ssa_progress_label = None
        self.ssa_progress_bar = None
        
        #sets flag to track if simulation is running
        self.simulation_running = True
        
        #starts the progress update loop
        self.update_progress()
        
        #runs the CLE simulation in a separate thread
        thread = threading.Thread(target=self.run_cle_simulation_thread, daemon=True)
        thread.start()
    
    #function that runs CLE simulation in a separate thread
    def run_cle_simulation_thread(self):
        try:
            self.cle_results = run_cle_simulation([self.k1, self.k2, self.k3, self.k4, self.A, self.B, self.cle_initial, self.T, self.N, self.M, self.random_seed, self.det_family, self.det_order, self.stoch_integrator, self.op_splitting], self.cle_progress_tracker)
            self.after(0, self.after_cle_simulation)
        except Exception as e: 
            error_msg = str(e)
            print(f"CLE simulation error: {error_msg}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda msg=error_msg: self.show_error(msg))
    
    #function that runs SSA simulation in a separate thread
    def run_ssa_simulation_thread(self):
        try:
            self.ssa_results = run_ssa_simulation([self.k1, self.k2, self.k3, self.k4, self.A, self.B, self.ssa_initial, self.T, self.N, self.M, self.random_seed], self.ssa_progress_tracker)
            self.after(0, self.after_ssa_simulation)
        except Exception as e: 
            error_msg = str(e)
            print(f"SSA simulation error: {error_msg}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda msg=error_msg: self.show_error(msg))
    
    #function that handles completion of CLE simulation
    def after_cle_simulation(self):
        #destroys CLE progress bar and label
        if hasattr(self, 'cle_progress_bar') and self.cle_progress_bar.winfo_exists(): self.cle_progress_bar.destroy()
        if hasattr(self, 'cle_progress_label') and self.cle_progress_label.winfo_exists(): self.cle_progress_label.destroy()
        
        #updates label to show creating visuals
        self.main_label.configure(text="Creating Visuals", font=("Times New Roman", 54))
        self.update()
        
        #creates CLE-only visualizations in a separate thread
        thread = threading.Thread(target=self.create_cle_visuals_thread, daemon=True)
        thread.start()
    
    #function that creates CLE-only visualizations in a separate thread
    def create_cle_visuals_thread(self):
        try:
            t_grid_cle, Y_all_cle = self.cle_results
            theme = ctk.get_appearance_mode().lower()
            output_dir = "graphs"
            cle_visuals.ensure_dir(output_dir)
            
            #gets solver name for bias summary 
            solver_name = f"{self.stoch_integrator} ({self.det_family} {self.det_order})"
            safe_name = solver_name.replace(' ', '_').replace('–', '-').replace('/', '_').replace('\\', '_')
            safe_name = safe_name.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
            safe_name = safe_name.replace('{', '').replace('}', '').replace(':', '_').replace('*', '_')
            safe_name = safe_name.replace('?', '').replace('"', '').replace('<', '').replace('>', '')
            safe_name = safe_name.replace('|', '_')
            self.solver_bias_filename = f"cle_solver_bias_{safe_name}.png"
            
            #defines the 7 CLE-only plots
            cle_plots = [
                (cle_visuals.plot_cle_time_series, (t_grid_cle, Y_all_cle, output_dir, theme)),
                (cle_visuals.plot_cle_mean_variance, (t_grid_cle, Y_all_cle, output_dir, theme)),
                (cle_visuals.plot_cle_stationary_distribution, (t_grid_cle, Y_all_cle, output_dir, theme)),
                (cle_visuals.plot_cle_quantiles, (t_grid_cle, Y_all_cle, output_dir, theme)),
                (cle_visuals.plot_cle_autocorrelation, (t_grid_cle, Y_all_cle, output_dir, theme)),
                (cle_visuals.plot_cle_switching_events, (t_grid_cle, Y_all_cle, output_dir, theme)),
                (cle_visuals.plot_cle_solver_bias_summary, (t_grid_cle, Y_all_cle, solver_name, output_dir, theme))
            ]
            
            #uses multiprocessing with 1 core to create plots
            with multiprocessing.Pool(processes=1) as pool:
                pool.map(create_plot_multiprocess, cle_plots)
            
            self.after(0, self.create_cle_plot_buttons)
            
            #creates SSA progress tracker and progress bar at the bottom
            self.after(0, self.setup_ssa_simulation)
            
        except Exception as e: 
            error_msg = str(e)
            print(f"CLE visuals error: {error_msg}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda msg=error_msg: self.show_error(msg))
    
    #function that creates buttons for CLE plots
    def create_cle_plot_buttons(self):
        #configures grid for buttons
        for i in range(50): self.grid_rowconfigure(i, weight=1)
        for j in range(4): self.grid_columnconfigure(j, weight=1)
        
        #defines the 7 CLE-only plots
        cle_plots = [
            ("Time Series", "cle_time_series.png", "CLE Trajectories"),
            ("Mean & Variance", "cle_mean_variance.png", "CLE Ensemble Statistics"),
            ("Stationary Distribution", "cle_stationary_distribution.png", "CLE Stationary Distribution"),
            ("Quantiles", "cle_quantiles.png", "CLE Quantile Analysis"),
            ("Autocorrelation", "cle_autocorrelation.png", "CLE Autocorrelation Function"),
            ("Switching Events", "cle_switching_events.png", "CLE Switching Events"),
            ("Solver Bias", self.solver_bias_filename, "CLE Solver Bias Summary")
        ]
        
        #creates buttons in a grid 
        for i, (button_text, filename, title) in enumerate(cle_plots):
            if i < 4:
                row = 10
                col = i
                colspan = 1
            else:
                row = 16 
                if i == 4:
                    col = 0
                    colspan = 1
                elif i == 5:
                    col = 1
                    colspan = 2
                else:  # i == 6
                    col = 3
                    colspan = 1
            btn = ctk.CTkButton(self, text=button_text, font=("Times New Roman", 18), command=lambda f=filename, t=title: self.open_plot_window(f, t))
            btn.grid(row=row, column=col, columnspan=colspan, padx=5, pady=5, sticky="nsew")
    
    #function that sets up SSA simulation
    def setup_ssa_simulation(self):
        self.main_label.configure(text="CLE simulation complete, running SSA\nthis may take a long time")
        
        #creates SSA progress tracker and progress bar at the very bottom
        self.ssa_progress_tracker = ProgressTracker()
        
        self.ssa_progress_label = ctk.CTkLabel(self, text="SSA Progress: 0/0", font=("Times New Roman", 28))
        self.ssa_progress_label.grid(row=47, column=0, columnspan=4, pady=5, sticky="swe")
        
        self.ssa_progress_bar = ctk.CTkProgressBar(self, orientation="horizontal", width=500, height=20, corner_radius=10, fg_color=("#ffffff", "#0d0d0d"), progress_color=("#ff8c42", "#ff8c42"))
        self.ssa_progress_bar.set(0)
        self.ssa_progress_bar.grid(row=48, column=0, columnspan=4, pady=5, sticky="swe")
        
        #starts SSA simulation in a separate thread
        thread = threading.Thread(target=self.run_ssa_simulation_thread, daemon=True)
        thread.start()
    
    #function that handles completion of SSA simulation
    def after_ssa_simulation(self):
        #destroys SSA progress bar and label
        if hasattr(self, 'ssa_progress_bar') and self.ssa_progress_bar.winfo_exists(): self.ssa_progress_bar.destroy()
        if hasattr(self, 'ssa_progress_label') and self.ssa_progress_label.winfo_exists(): self.ssa_progress_label.destroy()
        
        #updates label to show creating visuals
        self.main_label.configure(text="Creating Visuals", font=("Times New Roman", 54))
        self.update()
        
        #creates CLE vs SSA visualizations using multiprocessing
        thread = threading.Thread(target=self.create_comparison_visuals_thread, daemon=True)
        thread.start()
    
    #function that creates the comparison plots
    def create_comparison_visuals_thread(self):
        try:
            t_grid_cle, Y_all_cle = self.cle_results
            t_grid_ssa, Y_all_ssa = self.ssa_results
            theme = ctk.get_appearance_mode().lower()
            output_dir = "graphs"
            cle_visuals.ensure_dir(output_dir)
            
            #defines the 6 comparison plots
            comparison_plots = [
                (cle_visuals.plot_ssa_vs_cle_distributions, (t_grid_cle, Y_all_cle, t_grid_ssa, Y_all_ssa, output_dir, theme)),
                (cle_visuals.plot_ssa_vs_cle_quantiles, (t_grid_cle, Y_all_cle, t_grid_ssa, Y_all_ssa, output_dir, theme)),
                (cle_visuals.plot_ssa_vs_cle_bias, (t_grid_cle, Y_all_cle, t_grid_ssa, Y_all_ssa, output_dir, theme)),
                (cle_visuals.plot_probability_mass_regions, (t_grid_cle, Y_all_cle, t_grid_ssa, Y_all_ssa, output_dir, theme)),
                (cle_visuals.plot_switching_statistics, (t_grid_cle, Y_all_cle, t_grid_ssa, Y_all_ssa, output_dir, theme)),
                (cle_visuals.plot_time_resolved_comparison, (t_grid_cle, Y_all_cle, t_grid_ssa, Y_all_ssa, output_dir, theme))
            ]
            
            #uses multiprocessing with 1 core to create plots
            with multiprocessing.Pool(processes=1) as pool:
                pool.map(create_plot_multiprocess, comparison_plots)
            
            #creates buttons and finalizes
            self.after(0, self.finalize_simulation)
            
        except Exception as e: 
            error_msg = str(e)
            print(f"Comparison visuals error: {error_msg}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda msg=error_msg: self.show_error(msg))
    
    #function that finalizes simulation and creates all buttons
    def finalize_simulation(self):
        #stops the progress update loop
        self.simulation_running = False
        
        #updates main label
        self.main_label.configure(text="Simulation Results", font=("Times New Roman", 54))
        
        #configures grid for buttons
        for i in range(50): self.grid_rowconfigure(i, weight=1)
        for j in range(4): self.grid_columnconfigure(j, weight=1)
        
        #defines the 6 comparison plots
        comparison_plots = [
            ("SSA vs CLE Distributions", "ssa_vs_cle_distributions.png", "SSA vs CLE Stationary Distributions"),
            ("SSA vs CLE Quantiles", "ssa_vs_cle_quantiles.png", "SSA vs CLE Quantile Comparison"),
            ("SSA vs CLE Bias", "ssa_vs_cle_bias.png", "CLE Bias Relative to SSA"),
            ("Probability Mass Regions", "probability_mass_regions.png", "Probability Mass in Physical Regions"),
            ("Switching Statistics", "switching_statistics.png", "Switching Statistics: SSA vs CLE"),
            ("Time-Resolved Comparison", "time_resolved_comparison.png", "Time-Resolved Comparison: SSA vs CLE")
        ]
        
        #creates buttons for comparison plots
        for i, (button_text, filename, title) in enumerate(comparison_plots):
            if i < 4:
                row = 22  
                col = i
                colspan = 1
            else:
                row = 28  
                col = (i - 4) * 2  
                colspan = 2  
            btn = ctk.CTkButton(self, text=button_text, font=("Times New Roman", 18), command=lambda f=filename, t=title: self.open_plot_window(f, t))
            btn.grid(row=row, column=col, columnspan=colspan, padx=5, pady=5, sticky="nsew")
        
        #creates export, restart, and exit buttons at the bottom
        export_graphs_btn = ctk.CTkButton(self, text="Export All Graphs", font=("Times New Roman", 21), command=self.export_all_graphs)
        export_graphs_btn.grid(row=45, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        export_data_btn = ctk.CTkButton(self, text="Export All Data", font=("Times New Roman", 21), command=self.export_all_data)
        export_data_btn.grid(row=45, column=2, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        restart_btn = ctk.CTkButton(self, text="Restart", font=("Times New Roman", 21), command=self.restart_app)
        restart_btn.grid(row=46, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        exit_btn = ctk.CTkButton(self, text="Exit", font=("Times New Roman", 21), command=self.exit_app)
        exit_btn.grid(row=46, column=2, columnspan=2, sticky="nsew", padx=10, pady=10)
    
    #function that opens a plot window
    def open_plot_window(self, filename, title):
        img_path = os.path.join("graphs", filename)
        if not os.path.exists(img_path):
            self.main_label.configure(text=f"Graph not found: {filename}")
            return
        
        #loads the image to get its dimensions
        pil_img = Image.open(img_path)
        img_width, img_height = pil_img.size
        
        #calculates window width based on image width (maintains aspect ratio, max 1200px width)
        #keeps height at 520 as specified
        display_height = 500
        aspect_ratio = img_width / img_height
        display_width = min(int(display_height * aspect_ratio), 1200)
        window_width = display_width + 20  #add padding for window borders
        window_height = 520
        
        #creates and configures a top level window
        win = ctk.CTkToplevel(self)
        win.title(title)
        win.geometry(f"{window_width}x{window_height}")
        win.resizable(False, False)
        
        #resizes the image to fit the display
        img = CTkImage(light_image=pil_img, dark_image=pil_img, size=(display_width, display_height))
        panel = ctk.CTkLabel(win, image=img, text="")
        panel.image = img
        panel.pack(pady=5)
    
    #function that exports all graphs
    def export_all_graphs(self):
        output_dir = "graphs"
        if not os.path.exists(output_dir) or not os.listdir(output_dir):
            self.main_label.configure(text="No graphs to export")
            self.after(3000, lambda: self.main_label.configure(text="Simulation Results"))
            return
        
        #asks user to select the destination
        zip_path = filedialog.asksaveasfilename(defaultextension=".zip", initialfile="CLE_Graphs.zip", title="Save All Graphs", filetypes=[("ZIP Archive", "*.zip")])
        if not zip_path: return
        
        #exports all graphs to the selected zip file
        with zipfile.ZipFile(zip_path, 'w') as z:
            for file in os.listdir(output_dir):
                if file.endswith(".png"):
                    z.write(os.path.join(output_dir, file), arcname=file)
    
    #function that exports all data
    def export_all_data(self):
        #asks user to select the destination
        zip_path = filedialog.asksaveasfilename(defaultextension=".zip", initialfile="CLE_Data.zip", title="Save All Data", filetypes=[("ZIP Archive", "*.zip")])
        if not zip_path: return
        
        #makes a temporary data directory
        temp_dir = "temp_data_export"
        os.makedirs(temp_dir, exist_ok=True)
        
        #saves CLE results
        t_grid_cle, Y_all_cle = self.cle_results
        np.savez(os.path.join(temp_dir, "cle_results.npz"), t_grid=t_grid_cle, Y_all=Y_all_cle)
        
        #saves SSA results
        t_grid_ssa, Y_all_ssa = self.ssa_results
        np.savez(os.path.join(temp_dir, "ssa_results.npz"), t_grid=t_grid_ssa, Y_all=Y_all_ssa)
        
        #saves simulation parameters
        params = {
            "k1": self.k1, "k2": self.k2, "k3": self.k3, "k4": self.k4,
            "A": self.A, "B": self.B,
            "cle_initial": self.cle_initial, "ssa_initial": self.ssa_initial,
            "T": self.T, "N": self.N, "M": self.M, "random_seed": self.random_seed,
            "det_family": self.det_family, "det_order": self.det_order,
            "stoch_integrator": self.stoch_integrator, "op_splitting": self.op_splitting
        }
        np.savez(os.path.join(temp_dir, "simulation_parameters.npz"), **params)
        
        #saves all files to the selected path
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for file in os.listdir(temp_dir):
                zipf.write(os.path.join(temp_dir, file), arcname=file)
        
        #removes the temporary directory
        for file in os.listdir(temp_dir): os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

    #function that updates the progress bars
    def update_progress(self):
        try:
            #updates CLE progress
            if self.cle_progress_tracker is not None and hasattr(self, 'cle_progress_bar') and self.cle_progress_bar.winfo_exists():
                current_cle = self.cle_progress_tracker.current_step
                total_cle = self.cle_progress_tracker.total_steps
                if total_cle > 0:
                    self.cle_progress_bar.set(current_cle / total_cle)
                    self.cle_progress_label.configure(text=f"CLE Progress: {current_cle}/{total_cle}")
            
            #updates SSA progress
            if self.ssa_progress_tracker is not None and hasattr(self, 'ssa_progress_bar') and self.ssa_progress_bar.winfo_exists():
                current_ssa = self.ssa_progress_tracker.current_step
                total_ssa = self.ssa_progress_tracker.total_steps
                if total_ssa > 0:
                    self.ssa_progress_bar.set(current_ssa / total_ssa)
                    self.ssa_progress_label.configure(text=f"SSA Progress: {current_ssa}/{total_ssa}")
        except: pass
        
        #continues updating if simulation is running (either CLE or SSA)
        if self.simulation_running: self.after(200, self.update_progress)
    
    #function that displays an error message if the simulation failed
    def show_error(self, error_msg):
        #stops the progress update loop
        self.simulation_running = False
        
        #removes progress bars if they exist and are not None
        if hasattr(self, 'cle_progress_bar') and self.cle_progress_bar is not None:
            try:
                if self.cle_progress_bar.winfo_exists(): self.cle_progress_bar.grid_remove()
            except:
                pass
        if hasattr(self, 'ssa_progress_bar') and self.ssa_progress_bar is not None:
            try:
                if self.ssa_progress_bar.winfo_exists(): self.ssa_progress_bar.grid_remove()
            except:
                pass
        
        #configures grid for error display 
        for i in range(50): self.grid_rowconfigure(i, weight=1)
        for j in range(4): self.grid_columnconfigure(j, weight=1)
        
        #configures the main label with error message
        self.main_label.configure(text=f"Error running simulation:\n{error_msg}", font=("Times New Roman", 38))
        self.main_label.grid(row=20, rowspan=10, column=0, columnspan=4, pady=10, sticky="nsew")
        
        #creates buttons to 
        show_error_btn = ctk.CTkButton(self, text="Show Error Details", font=("Times New Roman", 21), command=lambda: self.show_error_details(error_msg))
        show_error_btn.grid(row=35, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        restart_btn = ctk.CTkButton(self, text="Restart", font=("Times New Roman", 21), command=self.restart_app)
        restart_btn.grid(row=35, column=2, columnspan=2, sticky="nsew", padx=10, pady=10)
    
    #function that shows error details in a popup
    def show_error_details(self, error_msg):
        #creates a popup window to show error details
        popup = ctk.CTkToplevel(self)
        popup.title("Error Details")
        popup.geometry("600x400")
        popup.resizable(True, True)
        
        #creates a text widget to display the error
        text_widget = ctk.CTkTextbox(popup, font=("Consolas", 12))
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)
        text_widget.insert("1.0", error_msg)
        text_widget.configure(state="disabled")
        
        #creates a close button
        close_btn = ctk.CTkButton(popup, text="Close", font=("Times New Roman", 18), command=popup.destroy)
        close_btn.pack(pady=10)
        
    #function that toggles the theme of the window
    def toggle_theme(self):
        current = ctk.get_appearance_mode().lower()
        ctk.set_appearance_mode("light") if current == "dark" else ctk.set_appearance_mode("dark")
        
    #function that restarts the program
    def restart_app(self):
        self.destroy()
        subprocess.call([sys.executable, sys.argv[0]])

    #function that exits the program
    def exit_app(self): self.destroy()

#begins the program
if __name__ == "__main__":
    app = CLEApp()
    app.mainloop()