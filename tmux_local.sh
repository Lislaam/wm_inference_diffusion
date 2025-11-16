#!/bin/bash

# Script for installing tmux on systems where you don't have root access.
# tmux will be installed in $INSTALL_DIR/local/bin.
# It's assumed that wget and a C/C++ compiler are installed.

# exit on error
set -e

TMUX_VERSION=1.8
INSTALL_DIR=/home/aguzel

# create our directories
mkdir -p $INSTALL_DIR/local $INSTALL_DIR/tmux_tmp
cd $INSTALL_DIR/tmux_tmp

# download source files for tmux, libevent, and ncurses
wget -O tmux-${TMUX_VERSION}.tar.gz http://sourceforge.net/projects/tmux/files/tmux/tmux-${TMUX_VERSION}/tmux-${TMUX_VERSION}.tar.gz/download
wget https://github.com/downloads/libevent/libevent/libevent-2.0.19-stable.tar.gz
wget ftp://ftp.gnu.org/gnu/ncurses/ncurses-5.9.tar.gz

# extract files, configure, and compile

############
# libevent #
############
tar xvzf libevent-2.0.19-stable.tar.gz
cd libevent-2.0.19-stable
./configure --prefix=$INSTALL_DIR/local --disable-shared
make
make install
cd ..

############
# ncurses  #
############
tar xvzf ncurses-5.9.tar.gz
cd ncurses-5.9
./configure --prefix=$INSTALL_DIR/local
make
make install
cd ..

############
# tmux     #
############
tar xvzf tmux-${TMUX_VERSION}.tar.gz
cd tmux-${TMUX_VERSION}
./configure CFLAGS="-I$INSTALL_DIR/local/include -I$INSTALL_DIR/local/include/ncurses" LDFLAGS="-L$INSTALL_DIR/local/lib -L$INSTALL_DIR/local/include/ncurses -L$INSTALL_DIR/local/include"
CPPFLAGS="-I$INSTALL_DIR/local/include -I$INSTALL_DIR/local/include/ncurses" LDFLAGS="-static -L$INSTALL_DIR/local/include -L$INSTALL_DIR/local/include/ncurses -L$INSTALL_DIR/local/lib" make
cp tmux $INSTALL_DIR/local/bin
cd ..

# cleanup
rm -rf $INSTALL_DIR/tmux_tmp

echo "$INSTALL_DIR/local/bin/tmux is now available. You can optionally add $HOME/local/bin to your PATH."