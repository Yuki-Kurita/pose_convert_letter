# coding:utf-8
from tkinter import *
import tkinter.ttk as ttk
import tkinter.messagebox as messagebox
import sys

class MenuButtonSample(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.create_widgets()
        self.pack()

    def create_widgets(self):
        menub = ttk.Menubutton(self,text='select address')
        menu  = Menu(menub)
        menub['menu']= menu
        menu.add_command(label="Level1",command =self.level_one)
        menu.add_command(label="Level2",command =self.level_two)
        menub.pack()

    def level_one(self):
        print("aa")

    def level_two(self):
        print("b")

if __name__ == '__main__':
    master = Tk()
    master.title("MenuButtonSample")
    master.geometry("300x100")
    MenuButtonSample(master)
    master.mainloop()
