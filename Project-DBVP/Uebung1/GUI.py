import tkinter
from tkinter import *
from tkinter import filedialog
import cv2
from PIL import ImageTk,Image
import numpy as np
from tkinter import messagebox

# global definition
global window
img = None 
img1 = None
canves1 = None
X =  np.empty((0,2), float)
A = np.empty((0,2), float)



def rotePunkte(eventorigin):
    global X
    if (X.size < 8):
        x = eventorigin.x
        y = eventorigin.y
        coordi_str = "x:"+str(x)+", "+"y:"+str(y)
        tkinter.Label(window, width=100, text = coordi_str).grid(row = 1, column = 2)
        print (x,y)
        global canves1
        canves1.create_oval(x+10, y+10, x-10, y-10, fill = "#FF0000")
        
        X = np.append(X, np.array([[x,y]],dtype=float), axis=0)
        print (X)
    else:
        messagebox.showinfo("Information","Sie haben schon genug ausgewählt!\nBitte Suchen Sie sich die anderen 4 Punkte mit Rechtsklick!")



def blauPunkte(eventorigin):
    global A
    if (A.size < 8 ):
        x = eventorigin.x
        y = eventorigin.y
        global canves1
        coordi_str = "x:"+str(x)+", "+"y:"+str(y)
        tkinter.Label(window, width=100, text = coordi_str).grid(row = 1, column = 2)
        print (x,y)
        canves1.create_oval(x+10, y+10, x-10, y-10, fill = "#0000FF")
        A = np.append(A, np.array([[x,y]],dtype=float),axis=0)
        print(A)
    else:
        messagebox.showinfo("Information","Sie haben schon genug ausgewählt!\nBitte Suchen Sie sich die anderen 4 Punkte mit linksklick!")


def UpOnAction(event=None):
    global img_resized
    global img1
    global A, X
    if (X.shape[0] != 4 or A.shape[0] != 4):
        messagebox.showinfo("Information","Sie haben schon keine ausreichenden Punkte ausgewählt!")
        return
    A=np.float32(A)
    X=np.float32(X)
    H = cv2.getPerspectiveTransform(X,A)
    out = cv2.warpPerspective(np.array(img_resized),H,(img_resized.size[0], img_resized.size[1]),flags=cv2.INTER_LINEAR)
    # plt.imshow(out)
    # plt.show()
    cv2.imwrite("result.jpg", out)
    # Display the transformed image
    img1 = Image.open("result.jpg")
    

    W1, H1 = img1.size
    H1 = (H1*800)/W1
    img_resized1=img1.resize((800,int(H1)))
    img1 = ImageTk.PhotoImage(img_resized1)
    global canves1
    canves1.create_image(0,0, image = img1, anchor = "nw")
    
    

def UploadAction(event=None):
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    global img
    global img_resized
    img=Image.open(filename)
    W, H = img.size
    H = (H*800)/W
    img_resized=img.resize((800,int(H)))
    img = ImageTk.PhotoImage(img_resized)
    global canves1
    canves1 = tkinter.Canvas(window, width=W,height=H)
    canves1.grid(row = 3, column = 2)
    canves1.create_image(0,0, image = img, anchor = "nw")
    canves1.bind("<Button 1>", rotePunkte)
    canves1.bind("<Button 3>", blauPunkte)

    

def restart(event=None):
    window.destroy()  
    programm()


# Main Programm
def programm():
    global window
    window = tkinter.Tk()
    window.title("Perespektive Transformation")
    window.geometry("1000x800")
    window.resizable(0, 0)
    tkinter.Button(window, text='Bild hochladen', command=UploadAction).grid(row = 1, column = 1)
    tkinter.Button(window, text='Tranformieren', command=UpOnAction).grid(row = 2, column = 1)
    tkinter.Button(window, text='Neu Bearbeitung', command=restart).grid(row = 3, column = 1)
    window.mainloop()

programm()