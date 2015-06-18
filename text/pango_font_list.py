#!/usr/bin/python

# Code adapted from the following links on the internet:
# http://zetcode.com/gui/pyGtk/pango/
#

from gi.repository import Gtk
from gi.repository import Pango as pango
import cairo
import math

class PyApp(Gtk.Window): 
    def __init__(self):
        super(PyApp, self).__init__()
        
        self.connect("destroy", Gtk.main_quit)
        self.set_title("Quotes")
        
        label = Gtk.Label("Anuj khare rules")
        #Gtk.Gdk.beep()

        fontdesc = pango.FontDescription("Purisa 10")
        label.modify_font(fontdesc)

        fix = Gtk.Fixed()

        fix.put(label, 5, 5)
        
        self.add(fix)
        #self.set_position(Gtk.WIN_POS_CENTER)
        self.show_all()

class PyApp1(Gtk.Window): 
    def __init__(self):
        super(PyApp1, self).__init__()
        
        self.set_size_request(350, 250)
        self.set_border_width(8)
        self.connect("destroy", Gtk.main_quit)
        self.set_title("System fonts")
        
        sw = Gtk.ScrolledWindow()
        #sw.set_shadow_type(Gtk.SHADOW_ETCHED_IN)
        #sw.set_policy(Gtk.POLICY_AUTOMATIC, Gtk.POLICY_AUTOMATIC)
        
        context = self.create_pango_context()
        self.fam = context.list_families()

        store = self.create_model()

        treeView = Gtk.TreeView(store)
        treeView.set_rules_hint(True)
        sw.add(treeView)

        self.create_column(treeView)

        self.add(sw)
        
        #self.set_position(Gtk.WIN_POS_CENTER)
        self.show_all()


    def create_column(self, treeView):
        rendererText = Gtk.CellRendererText()
        column = Gtk.TreeViewColumn("FontName", rendererText, text=0)
        column.set_sort_column_id(0)    
        treeView.append_column(column)
    
    def create_model(self):
        store = Gtk.ListStore(str)

        for ff in self.fam:
            store.append([ff.get_name()])
            print (ff.get_name())

        return store

PyApp1()
Gtk.main()
