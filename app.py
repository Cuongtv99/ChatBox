from tkinter import *
from chatbot import chatbot_response, bot_name


BG_GRAY = '#abb2b9'
BG_COLOR = '#17202a'
TEXT_COLOR = '#eaecee'

FONT = 'Helvetica 14'
FONT_BOLD = 'Helvetica 13 bold'


class ChatApplication:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title('Chat Bot')
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)

        # head label
        head_lable = Label(self.window, text="WELLCOME",
                           font=FONT_BOLD, bg=BG_COLOR, fg=TEXT_COLOR, pady=10)
        head_lable.place(relwidth=1)

        # tinny divider
        line = Label(self.window, width=450, bg=BG_GRAY, pady=10)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # scrollbar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        # bottom label
        bottom_lable = Label(self.window, bg=BG_GRAY, height=80)
        bottom_lable.place(relwidth=1, rely=0.825)

        # message entry box
        self.msg_entry = Entry(bottom_lable, width=20,
                               bg="#2c3e50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relheight=0.06, relwidth=0.74,
                             relx=0.011, rely=0.008)
        self.msg_entry.focus()
        self.msg_entry.bind('<Return>', self._on_enter_pressed)

        # send button
        send_button = Button(bottom_lable, text="Send", width=20, bg=BG_GRAY,
                             font=FONT_BOLD, command=lambda: self._on_enter_pressed(None))
        send_button.place(relheight=0.06, relwidth=0.22, relx=0.77, rely=0.008)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")

    def _insert_message(self, msg, sender):
        if not msg:
            return
        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        msg2 = f"{bot_name}: {chatbot_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)


if __name__ == "__main__":
    app = ChatApplication()
    app.run()
