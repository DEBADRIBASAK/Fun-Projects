import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.io.*;

class NotePad {
	JFrame f;
	TextArea t;
	JMenuBar mb;
	JMenu fl,vw,cl;
	JMenuItem n,s,o,bg,fg;
	JCheckBoxMenuItem b,i;
	FileOutputStream fout;
	BufferedOutputStream bout;
	FileInputStream fin;
	BufferedInputStream bin;
	String buff;
	JSlider js;
	Font fn;
	JFileChooser fc;
	File ff;
	public NotePad()
	{
		f = new JFrame("Debadri's Notepad");
		t = new TextArea();
		mb = new JMenuBar();
		fl = new JMenu("File");
		vw = new JMenu("View");
		cl = new JMenu("Color");
		n = new JMenuItem("New",KeyEvent.VK_S);
		s = new JMenuItem("Save");
		o = new JMenuItem("Open");
		o.setMnemonic(KeyEvent.VK_O);
		b = new JCheckBoxMenuItem("Bold");
		i = new JCheckBoxMenuItem("Italic");
		bg  = new JMenuItem("Background");
		fg = new JMenuItem("ForeGround");
		js = new JSlider(10,72);
		fc = new JFileChooser();
		//buff = new byte[100];
	}
	public void addAll()
	{
		f.setJMenuBar(mb);
		mb.add(fl);
		mb.add(vw);
		mb.add(cl);
		fl.add(n);
		fl.add(s);
		fl.add(o);
		vw.add(b);
		vw.add(i);
		cl.add(bg);
		cl.add(fg);
		f.add(t,BorderLayout.CENTER);
		f.add(js,BorderLayout.NORTH);
		bg.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent e)
			{
				Color c = JColorChooser.showDialog(f,"Select Background Color",Color.WHITE);
				t.setBackground(c);
			}
		});
		fg.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent e)
			{
				Color c = JColorChooser.showDialog(f,"Select Background Color",Color.WHITE);
				t.setForeground(c);
			}
		});
		js.addChangeListener(new ChangeListener()
		{
			public void stateChanged(ChangeEvent e)
			{
				if(b.getState())
				{
				fn = new Font("Verdana",Font.BOLD,js.getValue());
				}
				if(i.getState())
				{
					fn = new Font("Verdana",Font.ITALIC,js.getValue());
				}
				else
				{
					fn = new Font("Verdana",Font.PLAIN,js.getValue());
				}
				t.setFont(fn);
			}
		});
		n.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent ae)
			{
				String name = JOptionPane.showInputDialog(f,"Enter The File Name");
				try{
				fout = new FileOutputStream(name);
				f.setTitle(name);
				bout = new BufferedOutputStream(fout);
				}
				catch(Exception e){}
			}
		});
		s.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent ae)
			{
				try
				{
					if(t.getText().length()==0)
					{
						throw new ArithmeticException();
					}
					else
					{
						bout.write(t.getText().getBytes());
						t.setText("");
					}
				}
				catch(ArithmeticException e)
				{
					JOptionPane.showMessageDialog(f,"Enter Some Text","ERROR!",JOptionPane.WARNING_MESSAGE);
				}
				catch(NullPointerException e)
				{
					JOptionPane.showMessageDialog(f,"Create a new file","ERROR!",JOptionPane.WARNING_MESSAGE);
				}
				catch(IOException e){}
				try{
				f.setTitle("Debadri's NotePad");
				bout.close();
				fout.close();
				bout = null;
				fout = null;
				}
				catch(Exception e){}
			}
		});
		o.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent ae)
			{
				//String name = JOptionPane.showInputDialog(f,"Enter The File Name");
				int i = fc.showOpenDialog(f);
				if(i==JFileChooser.APPROVE_OPTION)
				{
					ff = fc.getSelectedFile();
				}
				try{
				fin = new FileInputStream(ff);
				//f.setTitle(name);
				bin = new BufferedInputStream(fin);
				buff = new String("");
				int k;
				while((k = bin.read())!=-1)
				{
					buff+=(char)k;
				}
				t.setText(buff);
				bin.close();
				fin.close();
				fout = new FileOutputStream(ff);
				bout = new BufferedOutputStream(fout);
				}
				catch(Exception e){}
			}
		});
				
	}
	public void initialize()
	{
		f.setSize(500,500);
		f.setVisible(true);
	}
	public static void main(String args[])
	{
		NotePad obj = new NotePad();
		obj.addAll();
		obj.initialize();
	}
}
