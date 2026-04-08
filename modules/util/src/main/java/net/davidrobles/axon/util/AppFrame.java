package net.davidrobles.axon.util;

import javax.swing.*;

public class AppFrame extends JFrame {
    private JPanel panel;

    public AppFrame(JPanel panel) {
        this(panel, "");
    }

    public AppFrame(JPanel panel, String title) {
        super(title);
        this.panel = panel;
        add(panel);
        pack();
        //        DRUtil.centerComponent(this);
        setVisible(true);
    }
}
