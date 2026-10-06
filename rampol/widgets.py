"""Label-like text that can be selected and copied.

ttk.Label text cannot be selected, so a position or a result could only be
retyped. CopyLabel is a read-only tk.Text that looks like a label and keeps
the label calls the window uses - configure(text=..., foreground=...) and
cget("text") - so it drops in where a label was. Select with the mouse
(double-click a word, triple-click a line), Ctrl+C copies the selection;
right-click copies the selection or, with none, the whole text.
"""
import tkinter as tk
from tkinter import ttk


class CopyLabel(tk.Text):
    def __init__(self, parent, text="", foreground="black", width=None,
                 wraplength=None, justify="left", font=None, **_ignored):
        bg = ttk.Style().lookup("TFrame", "background")
        if width is None:
            width = max(10, int((wraplength or 300) / 7))
        super().__init__(parent, height=1, width=width, wrap="word", relief="flat",
                         borderwidth=0, highlightthickness=0, padx=0, pady=0,
                         background=bg or "SystemButtonFace", cursor="xterm",
                         font=font or "TkDefaultFont", takefocus=0,
                         exportselection=True)
        self.tag_configure("all", justify=justify)
        self._text = ""
        self._fg = foreground
        self.bind("<Button-1>", lambda e: self.focus_set(), add="+")
        self.bind("<Control-c>", self._copy)
        self.bind("<Button-3>", self._copy_all)
        self.bind("<Configure>", lambda e: self._fit())
        self.configure(text=text, foreground=foreground)

    def configure(self, cnf=None, **kw):
        kw = dict(cnf or {}, **kw)
        text = kw.pop("text", None)
        fg = kw.pop("foreground", kw.pop("fg", None))
        for k in ("wraplength", "justify", "anchor"):
            kw.pop(k, None)
        if fg is not None:
            self._fg = fg
            kw["foreground"] = fg
        out = super().configure(**kw) if kw else None
        if text is not None:
            self._text = str(text)
            super().configure(state="normal")
            self.delete("1.0", "end")
            self.insert("1.0", self._text, "all")
            super().configure(state="disabled")
            self._fit()
        return out

    config = configure

    def cget(self, key):
        if key == "text":
            return self._text
        return super().cget(key)

    def __setitem__(self, key, value):
        self.configure(**{key: value})

    def _fit(self):
        """Height = the lines the text wraps to (at least 1). Before the widget
        is on screen Tk measures the wrap at 1 px wide (one character per
        line - a 40-character label asked for 480 px and squeezed the Stop
        button out of the window), so until then estimate from the width in
        characters."""
        n = None
        if self.winfo_ismapped() and self.winfo_width() > 20:
            try:
                n = self.count("1.0", "end", "displaylines")
                n = n[0] if isinstance(n, tuple) else n
            except tk.TclError:
                n = None
        if not n:
            w = max(int(super().cget("width")), 1)
            n = sum(max(1, -(-len(line) // w)) for line in self._text.split("\n"))
        n = max(1, int(n))
        if int(super().cget("height")) != n:
            super().configure(height=n)

    def _copy(self, _e=None):
        try:
            txt = self.get("sel.first", "sel.last")
        except tk.TclError:
            txt = self._text
        self.clipboard_clear()
        self.clipboard_append(txt)
        return "break"

    def _copy_all(self, _e=None):
        return self._copy()
