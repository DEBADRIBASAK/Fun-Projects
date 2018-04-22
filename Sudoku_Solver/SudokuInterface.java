import java.awt.*;
import javax.swing.*;
import java.awt.event.*;

public class SudokuInterface extends JFrame implements ActionListener{
	JTextField tf;
	JButton b,r;
	ColorLabel board[][];
	int x,y;
	JPanel p1,p2;
	SudokuSolver sv;
	int arr[][];
class ColorLabel extends JLabel implements MouseListener
{
	int a,b;
	public ColorLabel(int a,int b)
	{
		this.a = a;
		this.b = b;
		Font f = new Font("Comic Sans MS",Font.BOLD,45);
		setFont(f);
		setBackground(Color.white);
		setOpaque(true);
		setForeground(Color.black);
		addMouseListener(this);
	}
	public void mouseClicked(MouseEvent me)
	{
		x = a;
		y = b;
		setForeground(Color.red);
		setBackground(Color.yellow);
		setText(tf.getText());
		arr[x][y] = Integer.parseInt(this.getText());
	}
	public void mouseEntered(MouseEvent me){}
	public void mouseExited(MouseEvent me){}
	public void mouseReleased(MouseEvent me){}
	public void mousePressed(MouseEvent me){}
   }
	public SudokuInterface()
	{
		setTitle("Solve Sudoku in a second!!");
		b = new JButton("SOLVE");
		r = new JButton("RESET");
		tf = new JTextField();
		board = new ColorLabel[9][9];
		arr = new int[9][9];
		p1 = new JPanel();
		p2 = new JPanel();
		p1.setBackground(Color.black);
		p1.setLayout(new GridLayout(9,9,2,2));
		Container c = this.getContentPane();
		c.setBackground(Color.red);
		c.setLayout(new BorderLayout(5,5));
		p2.setLayout(new GridLayout(1,3));
		for(int i=0;i<9;i++)
		{
			for(int j=0;j<9;j++)
			{
				board[i][j] = new ColorLabel(i,j);
				p1.add(board[i][j]);
				arr[i][j] = 0;
			}
		}
		p2.add(tf);
		p2.add(b);
		p2.add(r);
		b.addActionListener(this);
		r.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent e)
			{
				reset();
			}
		});
		c.add(p1,BorderLayout.CENTER);
		c.add(p2,BorderLayout.NORTH);
		setSize(500,500);
		setVisible(true);
	}
	public void setAll()
	{
		for(int i=0;i<9;i++)
		{
			for(int j=0;j<9;j++)
			{
				board[i][j].setText((new Integer(arr[i][j]).toString()));
			}
		}
	}
	public void reset()
	{
		for(int i=0;i<9;i++)
		{
			for(int j=0;j<9;j++)
			{
				board[i][j].setText("");
				board[i][j].setBackground(Color.WHITE);
				board[i][j].setForeground(Color.black);
				arr[i][j] = 0;
			}
		}
	}
	public void actionPerformed(ActionEvent e)
	{
		sv = new SudokuSolver(arr);
		sv.solve(0,0);
		arr = sv.getArray();
		setAll();
	}
	public static void main(String args[])
	{
		new SudokuInterface();
	}
}