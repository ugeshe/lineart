from __future__ import print_function

# Interactive visualization command in Jupyter Lab
# !pip install ipywidgets

import cv2
import os

from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider
import ipywidgets as widgets

import numpy as np              # import NumPy library


# Plot the function:
import matplotlib.pyplot as plt

import math # for floor and ceiling

# Import plotly
# !pip install kaleido

import plotly
import plotly.graph_objects as go
import plotly.express as px

# Data frames using pandas
import pandas as pd

from IPython.display import HTML
from base64 import b64encode



def f(m, b):
  # Get the figure and axes handles
  fig, ax = plt.subplots()

  # X-axis
  minX  = -10 # minimum value for X
  maxX  = +10 # maximum value for Y
  xdata = range(minX, maxX+1, 1) # Integer points to plot
  plt.xlim(minX, maxX) # Set the x-limits for the plot
  fig.set_figwidth((maxX-minX)/2.0)

  # Y-axis
  minY  = -5  # minimum value for Y
  maxY  = +5  # maximum value for Y
  ydata = range(minY, maxY+1, 1) # Integer grid to display
  plt.ylim(minY, maxY) # Set the y-limits for the plot
  fig.set_figheight((maxY-minY)/2.0)


  # Define the grid over integer values:
  ax.set_xticks(xdata)
  ax.set_yticks(ydata)
  plt.grid(True, alpha=0.5) # Make opacity softer

  # Plot the two axes and add tick points:
  plt.axvline(x=0.0, ymin=minY, ymax=maxY) # x-axis
  plt.axhline(y=0.0, xmin=minX, xmax=maxX) # y-axis

  # Plot the integer tick points for x:
  tick_min = 0.5 - 0.25*1/(maxY - minY) # fraction for min Y
  tick_max = 0.5 + 0.25*1/(maxY - minY) # fraction for max Y
  for x_tick in range(minX, 0, 1):
    plt.axvline(x=x_tick, ymin=tick_min, ymax=tick_max)
    plt.text(x=x_tick-0.25, y=-0.75, s=str(x_tick), fontsize=12)

  for x_tick in range(1, maxX+1, 1):
    plt.axvline(x=x_tick, ymin=tick_min, ymax=tick_max)
    plt.text(x=x_tick-0.25, y=-0.75, s=str(x_tick), fontsize=12)

  plt.text(x=-0.5, y=-0.5, s="0", fontsize=12)
  plt.text(x=maxX/2.0, y=-1.5, s="x", fontsize=12)

  # Plot the integer tick points for y:
  tick_min = 0.5 - 0.25*1/(maxX - minX) # fraction for min X
  tick_max = 0.5 + 0.25*1/(maxX - minX) # fraction for max X
  for y_tick in range(minY, 0, 1):
    plt.axhline(y=y_tick, xmin=tick_min, xmax=tick_max)
    plt.text(y=y_tick-0.25, x=-1.0, s=str(y_tick), fontsize=12)

  for y_tick in range(1, maxY+1, 1):
    plt.axhline(y=y_tick, xmin=tick_min, xmax=tick_max)
    plt.text(y=y_tick-0.25, x=-1.0, s=str(y_tick), fontsize=12)

  plt.text(y=maxY/2.0, x=-2.0, s="y", fontsize=12)

  # remove the outside tick axes:
  ax.set_xticklabels([])
  ax.set_yticklabels([])


  # Plot the points:
  for x in xdata:
    y = m*x + b
    plt.plot(x, y, "ro")
  yall = m*np.linspace(minX, maxX, maxX-minX+1)+b
  plt.plot(xdata, yall, color="red")

  # Equation legend:
  plt.legend([str("y = "+str(m)+"*x+"+str(b))], edgecolor="red")

  # Manually set the legend color
  leg = ax.get_legend()
  leg.legendHandles[0].set_color('red')

  # Set the figure size:
  # plt.figure(figsize=(15, 15))
  plt.title("y=m*x+b", fontsize=14)

  plt.show()


# Create the interactive visualization code class.
import collections
class cuteGraph:
    """
    The cuteGraph class is used for graphing linear functions:
        y = m*x + b
    The class uses integer limits for x.

    Common Attributes
    -----------------
    m: the slope of the line to plot.
    b: the y-intercept of the line for x=0.

    minX: minimum value for X. Default=-10.
    maxX: maximum value for X. Default=+10.

    minY: minimum value for Y. Default=-5.
    maxY: maximum value for Y. Default=+5.

    gridDisplay:   True if displaying the grid. False if not.
    legendDisplay: True if displaying the legend. False if not.


    color: "red", "green", "blue", "olive", "cyan", "brown", etc.
    See colors at the website below:
    https://matplotlib.org/3.1.0/gallery/color/named_colors.html


    Common Methods
    --------------
    plotGraph():           Builds a graph with the defaults.
    setX(minX, maxX):      Sets the integer bounds for X.
    setY(minY, maxY):      Sets the integer bounds for Y.

    line(m, b, color):  Adds a line y=m*x+b.
    point(x, y, color): Adds a point (x, y).

    lineSeg(x1, y1, x2, y2, color):   Adds line segment (x1, y1) to (x2, y2)
    rectangle(x1, y1, x2, y2, color): Add rectange with corners (x1, y1), (x2, y2).

    plotAll(): Generate the plot for everything.
    """

    def __init__(self):
        """ Creates a graph object with default parameters.
        """

        # Create the named tuples data structures:
        self.LineVal    = collections.namedtuple('line',  ['m', 'b', 'color'])
        self.HLineVal   = collections.namedtuple('hline', ['yvalue', 'color'])
        self.PointVal   = collections.namedtuple('point', ['x', 'y', 'color'])
        self.RectVal    = collections.namedtuple('rect',     ['x1', 'y1', 'x2', 'y2', 'color'])
        self.LineSegVal = collections.namedtuple('lineseg',  ['x1', 'y1', 'x2', 'y2', 'color'])

        # Create the lists:
        self.Lines    = []
        self.HLines   = []
        self.Points   = []
        self.Rects    = []
        self.LineSegs = []

        # Line and point widths:
        self.line_width  = 2
        self.point_width = 2

        # X-axis, Y-axis: use integer values.
        self.setStep(step=1, stepPixels=20)
        self.setX (minX=-10, maxX=+10)
        self.setY (minY=-5, maxY=+5)

        # Setup the grid, legend, and axis visibility
        self.gridDisplay   = True
        self.legendDisplay = True
        self.axisVis = True

        # Setup everything for plotly:
        self.fig = go.Figure()

    # Setup a square grid
    def setStep(self, step, stepPixels):
      self.step = step
      self.stepPixels = stepPixels

    # Grid options
    def gridOn(self):
      self.gridDisplay = True

    def gridOff(self):
      self.gridDisplay = False
      
    def prepVideo(self, magFactor):
      
      Dx = magFactor*(self.maxX - self.minX)
      Dy = magFactor*(self.maxY - self.minY)
      
      centerX = (self.maxX + self.minX)/2.0
      centerY = (self.maxY + self.minY)/2.0
      
      self.minX = centerX - Dx/2.0
      self.maxX = centerX + Dx/2.0
      self.minY = centerY - Dx/2.0
      self.maxY = centerY + Dx/2.0
      
      # Leave enough space on the right hand side for the text!
      self.setStep(step=1.0, stepPixels=20)
      
      # Turn off the extras before setting up the grid:
      self.allOff()
      
      # Prepare the grid
      self.setupGrid()
      
      # Main house
      self.setwidths(linewidth=5, pointwidth=2)


    # Legend options
    def legendOn(self):
      self.legendDisplay = True

    def legendOff(self):
      self.legendDisplay = False

    def axisOff(self):
      self.axisVis = False

    def allOff(self):
      self.gridOff()
      self.legendOff()
      self.axisOff()

    def addText(self, x, y, text, color):
      self.fig.add_annotation(
          x=x, y=y,
          text=text,  
          showarrow=False,
          align="center",
          font=dict(
            family="Courier New, monospace",
            size=16,
            color=color
            )
          )


    # .png, .jpeg, .svg, .pdf
    def saveImage(self, filename):
      """ saveImage(filename) saves filename.ext where ext=png, jpeg, svg, pdf
      """
      self.fig.write_image(filename, engine="kaleido", scale=1.0)

    # String representation for print() and debugging
    def __repr__(self) -> str:
      str_rep  = "CuteGraph class\n"
      str_rep += "MinX, MaxX, MinY, MaxY = "+str(self.minX)+", "+str(self.maxX)
      str_rep += ", "+str(self.minY)+", "+str(self.maxY)+"\n"
      str_rep += "Pixels width, height = "+str(self.width)+", "+str(self.height)
      str_rep += "Lines:"+str(self.Lines)+"\n"
      str_rep += "Horizontal Lines:"+str(self.HLines)+"\n"
      str_rep += "Points:"+str(self.Points)+"\n"
      str_rep += "Rectangles:"+str(self.Rects)+"\n"
      str_rep += "Line segments:"+str(self.LineSegs)+"\n"
      return str_rep

    def setX(self, minX, maxX):
      """ sets minimum and maximum values for x.
      """
      width = (maxX - minX + self.step) / self.step * self.stepPixels
      if (width <= 2048):
        self.minX  = minX
        self.maxX  = maxX
        self.width = width
      else:
        print("ERROR in setX(): too many pixels!")
        print("stepPixels="+str(self.stepPixels)+" is too large!")
        print("Width="+str(width)+" pixels > 1024")


    def setY(self, minY, maxY):
      """ sets minimum and maximum integer values for Y.
      """
      height = (maxY - minY + self.step) / self.step * self.stepPixels
      if (height <= 2048):
        self.minY  = minY
        self.maxY  = maxY
        self.height = height
      else:
        print("ERROR in setY(): too many pixels!")
        print("stepPixels="+str(self.stepPixels)+" is too large.")
        print("height="+str(height)+" pixels > 1024")


    def line(self, m, b, color):
      """ Adds a line with slope=m and y-intercept=b.
          Uses color to plot the line.
      """
      self.Lines.append(self.LineVal(m=m, b=b, color=color))

    def hline(self, yvalue, color):
      """ Adds a horizontal line with y=yvalue.
          Uses color to plot the line.
      """
      self.HLines.append(self.HLineVal(yvalue=yvalue, color=color))

    def point(self, x, y, color):
      """ Add a point with coordinates (x, y).
          Uses color to plot the point.
      """
      self.Points.append(self.PointVal(x=x,y=y, color=color))

    def rect(self, x1, y1, x2, y2, color):
      """ Add a rectangle with coordinates (x1, y1) and (x2, y2).
          Uses color to plot the point.
      """
      self.Rects.append(self.RectVal(x1=x1, y1=y1, x2=x2, y2=y2, color=color))

    def lineseg(self, x1, y1, x2, y2, color):
      """ Add a line segment with points (x1, y1) and (x2, y2).
          Uses color to plot the point.
      """
      self.LineSegs.append(self.LineSegVal(x1=x1, y1=y1, x2=x2, y2=y2, color=color))

    def setwidths(self, linewidth, pointwidth):
      self.line_width  = linewidth
      self.point_width = pointwidth

    def setupGrid(self, xdigits=3, xdecimals=0, ydigits=3, ydecimals=0, grid_color='red'):
      """ builds the grid using grid_color.
      """

      # X-axis
      minX = self.minX
      maxX = self.maxX
      numSamples = int((maxX-minX)/self.step)+1
      self.xdata = np.linspace(minX, maxX, numSamples)

      # Y-axis
      minY = self.minY
      maxY = self.maxY
      numSamples = int((maxY-minY)/self.step)+1
      self.ydata = np.linspace(minY, maxY, numSamples)

      # Plotly express figure ranges with grid:
      self.fig.update_yaxes(range=[minY, maxY],
                            showgrid=self.gridDisplay,
                            gridwidth=1,
                            gridcolor=grid_color,
                            scaleanchor = "x", scaleratio = 1,
                            automargin=False)
      self.fig.update_xaxes(range=[minX, maxX],
                            showgrid=self.gridDisplay,
                            gridwidth=1,
                            gridcolor=grid_color,
                            automargin=False)

      # Use plotly express layout for updates:
      self.fig.update_layout(xaxis = dict(
                              tickmode = 'linear',
                              tick0 = minX,
                              dtick = self.step
                                ),
                             yaxis = dict(
                                tickmode = 'linear',
                                tick0 = minY,
                                dtick = self.step
                                ),
                             height=self.height,
                             width=self.width,
                             autosize=False
                             )

      # Plot the two axes and add tick points:
      if (self.axisVis):
        self.fig.add_hline(y=0.0) # x-axis
        self.fig.add_vline(x=0.0) # y-axis

      # Update
      # self.fig.show()

    def plotAll(self):
      """ plots all defined lines and points.
      """

      # Plot Everything:
      for i in range(len(self.Points)):
        self.plotPoint(self.Points[i])

      for i in range(len(self.Lines)):
        self.plotLine(self.Lines[i])

      for i in range(len(self.Rects)):
        self.plotRect(self.Rects[i])

      for i in range(len(self.LineSegs)):
        self.plotLineSeg(self.LineSegs[i])

      for i in range(len(self.HLines)):
        self.plotHLine(self.HLines[i])

      # Setup the title for all of them:
      if (self.axisVis):
        self.fig.update_layout(
          title="Plots",
          xaxis_title="X",
          yaxis_title="Y",
          legend_title="List",
          font=dict(
            family="Courier New, monospace",
            size=12,
            color="RebeccaPurple"
          ))

      # Update the legend:
      self.fig.update_layout(showlegend=self.legendDisplay)

      # x axis visibility
      self.fig.update_xaxes(visible=self.axisVis)

      # y axis visibility
      self.fig.update_yaxes(visible=self.axisVis)

      # Update the graph:
      self.fig.show()

    def plotPoint(self, PointVal):
      """ plots a single point (x, y) using line_color
      """
      x_point, y_point, color = PointVal.x, PointVal.y, PointVal.color
      
      self.updateBounds(x_vals=[x_point], y_vals=[y_point])
      
      self.fig.add_trace(go.Scatter(
          x=[x_point], y=[y_point],
          marker=dict(size=18,
                      color=color,
                      line=dict(width=self.point_width,
                                color=color)),
          marker_color=color,
          marker_symbol="x",
          name="Point ("+str(x_point)+" ,"+str(y_point)+")",
          mode="markers"))

    def plotLine(self, LineVal):
      """ plots a single line given m, b, and line_color
      """
      m, b, line_color = LineVal.m, LineVal.b, LineVal.color

      # Do not support zoom-in:
      numpoints = 2
      xall = np.linspace(self.minX, self.maxX, numpoints)
      yall = m*xall + b

      # Use a simple name for the plot:
      if b>0:
        name_str = str("y = "+str(m)+"*x+"+str(b))
      elif b==0:
        name_str = str("y = "+str(m)+"*x")
      else:
        name_str = str("y = "+str(m)+"*x"+str(b))

      # Add the line plot:
      self.fig.add_trace(go.Scatter(
          x=xall,
          y=yall,
          mode="lines+markers",
          name=name_str,
          line=dict(color=line_color, width=self.line_width)))

    def plotHLine(self, HLineVal):
      """ plots a horizontal line y=yval using color
      """
      yvalue, color = HLineVal.yvalue, HLineVal.color
      self.fig.add_hline(y=yvalue,
          name="y="+str(yvalue),
          line_color=color, line_width=self.line_width)

    def plotLineSeg(self, LineSegVal):
      """ plots a line segment between two points
      """
      x1, x2, y1, y2 = LineSegVal.x1, LineSegVal.x2, LineSegVal.y1, LineSegVal.y2
      line_color = LineSegVal.color

      xall = [x1, x2]
      yall = [y1, y2]
      
      self.updateBounds(x_vals=xall, y_vals=yall)

      # Add the line plot:
      name_str  = "Line segment ("
      name_str += str(x1)+", "+str(y1)
      name_str += ") to ("+str(x2)+", "+str(y2)+")"
      self.fig.add_trace(go.Scatter(
          x=xall,
          y=yall,
          mode="lines+markers",
          name=name_str,
          line=dict(color=line_color, width=self.line_width)))
      
    def updateBounds(self, x_vals, y_vals):
      self.minX = min(self.minX, min(x_vals))
      self.maxX = max(self.maxX, max(x_vals))
      self.minY = min(self.minY, min(y_vals))
      self.maxY = max(self.maxY, max(y_vals))

    def plotRect(self, RectVal):
      """ plots a rectangle given two corner points.
      """
      x1, x2, y1, y2 = RectVal.x1, RectVal.x2, RectVal.y1, RectVal.y2
      line_color = RectVal.color
      
      # Update min and max
      self.updateBounds(x_vals=[x1, x2], y_vals=[y1, y2])

      name_str  = "Rect: line ("
      name_str += str(x1)+", "+str(y1)
      name_str += ") to ("+str(x1)+", "+str(y2)+")"
      self.fig.add_trace(go.Scatter(
          x=[x1, x1],
          y=[y1, y2],
          mode="lines+markers",
          name=name_str,
          line=dict(color=line_color, width=self.line_width)))

      name_str  = "Rect: line ("
      name_str += str(x2)+", "+str(y1)
      name_str += ") to ("+str(x2)+", "+str(y2)+")"
      self.fig.add_trace(go.Scatter(
          x=[x2, x2],
          y=[y1, y2],
          mode="lines+markers",
          name=name_str,
          line=dict(color=line_color, width=self.line_width)))

      name_str  = "Rect: line ("
      name_str += str(x1)+", "+str(y1)
      name_str += ") to ("+str(x2)+", "+str(y1)+")"
      self.fig.add_trace(go.Scatter(
          x=[x1, x2],
          y=[y1, y1],
          mode="lines+markers",
          name=name_str,
          line=dict(color=line_color, width=self.line_width)))

      name_str  = "Rect: line ("
      name_str += str(x1)+", "+str(y2)
      name_str += ") to ("+str(x2)+", "+str(y2)+")"
      self.fig.add_trace(go.Scatter(
          x=[x1, x2],
          y=[y2, y2],
          mode="lines+markers",
          name=name_str,
          line=dict(color=line_color, width=self.line_width)))


from skimage import io

def CreateVideo(video_name, file_list, fps):
  #self.plot_to_frame()
  height_list  = []
  width_list   = []
  video_frames = []

  # Find the biggest sizes
  for frame in file_list:
    img = io.imread(frame)
    video_frames.append(img)
    height, width, channels = img.shape
    height_list.append(height)
    width_list.append(width)

  #print(height_list)
  h_video=np.max(height_list)
  w_video=np.max(width_list)

  #fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
  video = cv2.VideoWriter(video_name,
            cv2.VideoWriter_fourcc(*'MJPG'),
            fps,
            (w_video, h_video))

  for frame in video_frames:
    video = padding(frame, video, h_video, w_video)

  video.release()

  vid = MakeVideo(video_name, width=w_video, height=h_video)
  return vid.HTML_str


def padding(frame, video, h_video, w_video):
  old_h, old_w, channels = frame.shape
  if divmod(h_video - old_h, 2)[1] != 0:
    pad_h_t = int((h_video - 1 - old_h) /2)
    pad_h_b = int((h_video + 1 - old_h) /2)

  else:
    pad_h_t = int((h_video - old_h) /2)
    pad_h_b = int((h_video - old_h) /2)

  if divmod(w_video - old_w, 2)[1] != 0:
    pad_w_l = int((w_video - 1 - old_w) /2)
    pad_w_r = int((w_video + 1 - old_w) /2)

  else:
    pad_w_l = int((w_video - old_w) /2)
    pad_w_r = int((w_video - old_w) /2)

  padding_image = cv2.copyMakeBorder(frame,
    pad_h_t, pad_h_b,
    pad_w_l, pad_w_r,
    cv2.BORDER_CONSTANT,
    None,
    value = [255, 255, 255])

  new_h, new_w, channels = padding_image.shape

  #plt.imshow(padding_image)
  #plt.show()
  padding_image = cv2.cvtColor(padding_image, cv2.COLOR_BGR2RGB)
  video.write(padding_image)
  return video


class MakeVideo:
  def __init__(self, mp4_fname, width, height):
    """ compresses the video file and displays in Jupyter notebook.
        mp4_fname represents the filename of the mp4 video.
    """
    self.mp4_fname = mp4_fname
    self.width  = width
    self.height = height

    # Compress file
    temp_file = "temp_video.mp4"
    self.compress(temp_file)

    # Create HTML for video display
    self.HTML_vid(temp_file)

  def compress(self, compressed_fname):
    """ compresses the given video file to compressed_finame
        If the filename is found, it replaces it with the current filename.
    """
    # Remove the compressed file name:
    if os.path.exists(compressed_fname):
      os.remove(compressed_fname)

    # Use ffmpeg to compress
    mp4_str = f"ffmpeg -i {self.mp4_fname} -vcodec libx264 {compressed_fname}"
    os.system(mp4_str)
    print("Compressed "+ self.mp4_fname + " into " + compressed_fname)

  def HTML_vid(self, compressed_fname):
    """ displays the compressed file in Juyter notebook.
    """
    mp4 = open(compressed_fname,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    self.HTML_str = """
                 <video width="%s" height="%s" controls loop autoplay>
                    <source src="%s" type="video/mp4">
                 </video> """ % (self.width, self.height, data_url)
