
import matplotlib.pyplot as plt
import numpy as np
import cv2


def display_four_points(image_filename, pts_):

  if len(pts_):

    fig_2, ax_2 = plt.subplots()
    ax_2.imshow(plt.imread(image_filename))
    ax_2.plot([pts_[0][0]], [pts_[0][1]], marker='o', markersize=3, color="red")
    ax_2.plot([pts_[1][0]], [pts_[1][1]], marker='o', markersize=3, color="red")
    ax_2.plot([pts_[2][0]], [pts_[2][1]], marker='o', markersize=3, color="red")
    ax_2.plot([pts_[3][0]], [pts_[3][1]], marker='o', markersize=3, color="red")
    
    plt.show()
    


def select_four_points(image_filename, pts_list):
  fig, ax = plt.subplots()
  ax.imshow(plt.imread(image_filename))
  
  def onclick(event):
    ix, iy = event.xdata, event.ydata
    # print(ix, iy)
    pts_list.append([ix, iy])

    # print(pts_list, len(pts_list))

    if len(pts_list) >= 4:
      fig.canvas.mpl_disconnect(cid)
      return

  cid = fig.canvas.mpl_connect('button_press_event', onclick)
  

def display_wraped_image(img):

  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  fig_3, ax_3 = plt.subplots()

  # Display the image
  ax_3.imshow(img)
  plt.show()


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


def four_point_transform(image_filename, pts):

  pts = np.array(pts)
  image = cv2.imread(image_filename)

  # obtain a consistent order of the points and unpack them
  # individually

  rect = order_points(pts)
  (tl, tr, br, bl) = rect
  # compute the width of the new image, which will be the
  # maximum distance between bottom-right and bottom-left
  # x-coordiates or the top-right and top-left x-coordinates
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))
  # compute the height of the new image, which will be the
  # maximum distance between the top-right and bottom-right
  # y-coordinates or the top-left and bottom-left y-coordinates
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))
  # now that we have the dimensions of the new image, construct
  # the set of destination points to obtain a "birds eye view",
  # (i.e. top-down view) of the image, again specifying points
  # in the top-left, top-right, bottom-right, and bottom-left
  # order
  dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
  # compute the perspective transform matrix and then apply it
  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
  # return the warped image
  return warped

