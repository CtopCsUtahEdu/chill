/*****************************************************************************
 Copyright (C) 2011 Chun Chen
 All Rights Reserved.

 Purpose:
   support command line editing for calculator.

 Notes:
   Since terminfo database is not queried for those nagging escape sequences,
 current supported terminials are limited to xterm, linux, cygwin.

 History:
   02/06/11 created by Chun Chen
*****************************************************************************/

#include <omega_calc/myflex.h>
#include <basic/util.h>
#include <string.h>
#include <stdlib.h>

#if defined __USE_POSIX
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#elif defined  __WIN32
#else
#endif

#define HISTORY_SIZE 100

namespace {
enum {MY_KEY_EOF=0, MY_KEY_LEFT, MY_KEY_RIGHT, MY_KEY_UP, MY_KEY_DOWN,
      MY_KEY_DEL, MY_KEY_HOME, MY_KEY_END, MY_KEY_PGUP, MY_KEY_PGDOWN,
      MY_KEY_NUMPAD_HOME, MY_KEY_NUMPAD_END};
}

extern bool is_interactive;
extern const char * PROMPT_STRING;

void move_cursor(int col, int n, int num_cols, const std::vector<std::string> &key_seqs) {
  if (n == 0)
    return;
  
  int new_col = omega::int_mod(col + n, num_cols);
  if (new_col == 0)
    new_col = num_cols;

  for (int i = 0; i < new_col-col; i++)
    std::cout.write(key_seqs[MY_KEY_RIGHT].c_str(), key_seqs[MY_KEY_RIGHT].length());
  for (int i = 0; i < col-new_col; i++)
    std::cout.write(key_seqs[MY_KEY_LEFT].c_str(), key_seqs[MY_KEY_LEFT].length());

  if (n < 0)
    for (int i = 0; i < omega::abs(n) / num_cols + (new_col>col)?1:0; i++)
      std::cout.write(key_seqs[MY_KEY_UP].c_str(), key_seqs[MY_KEY_UP].length());
  else
    for (int i = 0; i < omega::abs(n) / num_cols + (new_col<col)?1:0; i++)
      std::cout.write(key_seqs[MY_KEY_DOWN].c_str(), key_seqs[MY_KEY_DOWN].length());
}



myFlexLexer::myFlexLexer(std::istream *arg_yyin, std::ostream *arg_yyout):
  yyFlexLexer(arg_yyin, arg_yyout), history(HISTORY_SIZE), key_seqs(12) {
  cur_pos = 0;
  first_history_pos = 0;
  last_history_pos = -1;

  if (strcmp(getenv("TERM"), "xterm") == 0 ||
      strcmp(getenv("TERM"), "xterm-color") == 0) {
    key_seqs[MY_KEY_EOF] = "\x04";
    key_seqs[MY_KEY_LEFT] = "\x1B\x5B\x44";
    key_seqs[MY_KEY_RIGHT] = "\x1B\x5B\x43";
    key_seqs[MY_KEY_UP] = "\x1B\x5B\x41";
    key_seqs[MY_KEY_DOWN] = "\x1B\x5B\x42";
    key_seqs[MY_KEY_DEL] = "\x1B\x5B\x33\x7E";
    key_seqs[MY_KEY_HOME] = "\x1B\x4F\x48";
    key_seqs[MY_KEY_END] = "\x1B\x4F\x46";
    key_seqs[MY_KEY_PGUP] = "\x1B\x5B\x35\x7E";
    key_seqs[MY_KEY_PGDOWN] = "\x1B\x5B\x36\x7E";
    key_seqs[MY_KEY_NUMPAD_HOME] = "\x1B\x5B\x31\x7E";
    key_seqs[MY_KEY_NUMPAD_END] = "\x1B\x5B\x34\x7E";
  }
  else if (strcmp(getenv("TERM"), "linux") == 0 ||
           strcmp(getenv("TERM"), "cygwin") == 0) {
    key_seqs[MY_KEY_EOF] = "\x04";
    key_seqs[MY_KEY_LEFT] = "\x1B\x5B\x44";
    key_seqs[MY_KEY_RIGHT] = "\x1B\x5B\x43";
    key_seqs[MY_KEY_UP] = "\x1B\x5B\x41";
    key_seqs[MY_KEY_DOWN] = "\x1B\x5B\x42";
    key_seqs[MY_KEY_DEL] = "\x1B\x5B\x33\x7E";
    key_seqs[MY_KEY_HOME] = "\x1B\x5B\x31\x7E";
    key_seqs[MY_KEY_END] = "\x1B\x5B\x34\x7E";
    key_seqs[MY_KEY_PGUP] = "\x1B\x5B\x35\x7E";
    key_seqs[MY_KEY_PGDOWN] = "\x1B\x5B\x36\x7E";
    key_seqs[MY_KEY_NUMPAD_HOME] = "\x1B\x5B\x31\x7E";
    key_seqs[MY_KEY_NUMPAD_END] = "\x1B\x5B\x34\x7E";
  }
  else {
    key_seqs[MY_KEY_EOF] = "\x04";
  }
}

int myFlexLexer::LexerInput(char *buf, int max_size) {
  if (!is_interactive)
    return yyFlexLexer::LexerInput(buf, max_size);

#if defined __USE_POSIX
  winsize wsz;
  ioctl(0, TIOCGWINSZ, &wsz);
  int num_cols = wsz.ws_col;

  // unknown screen size, bail out
  if (num_cols == 0)
    return yyFlexLexer::LexerInput(buf, max_size);

  termios old_settings;
  termios new_settings;
  char keycodes[255];

  // set console to no echo, raw input mode
  tcgetattr(STDIN_FILENO, &old_settings);
  new_settings = old_settings;
  new_settings.c_cc[VTIME] = 1;
  new_settings.c_cc[VMIN] = 1;
  new_settings.c_iflag &= ~(IXOFF);
  new_settings.c_lflag &= ~(ECHO|ICANON);
  tcsetattr(STDIN_FILENO, TCSANOW, &new_settings);

  int cur_history_pos = (last_history_pos+1)%HISTORY_SIZE;
  while (true) {
    // feed current line to lex
    int len = cur_line.length();
    if (len > 0 && cur_line[len-1] == '\n') {
      int n = omega::min(len-cur_pos, max_size);
      for (int i = 0; i < n; i++)
        buf[i] = cur_line[cur_pos+i];
      cur_pos = cur_pos + n;
      if (cur_pos == len) {
        // save history
        if (len > 1) {
          if (last_history_pos == -1)
            last_history_pos = 0;
          else {
            last_history_pos = (last_history_pos+1)%HISTORY_SIZE;
            if ((last_history_pos + 1)%HISTORY_SIZE == first_history_pos)
              first_history_pos = (first_history_pos+1)%HISTORY_SIZE;
          }
          history[last_history_pos] = cur_line.substr(0, len-1);
          cur_history_pos = (last_history_pos+1)%HISTORY_SIZE;
        }

        // clear the working line
        cur_pos = 0;
        cur_line.clear();
      }      
      tcsetattr(STDIN_FILENO, TCSANOW, &old_settings);
      return n;
    }

    int count = read(STDIN_FILENO, keycodes, 255);
      
    // handle special key my way
    int eaten = 0;
    while (eaten <  count) {
      if (key_seqs[MY_KEY_EOF].length() > 0 &&
          static_cast<size_t>(count - eaten) >= key_seqs[MY_KEY_EOF].length() &&
          strncmp(&keycodes[eaten], key_seqs[MY_KEY_EOF].c_str(), key_seqs[MY_KEY_EOF].length()) == 0) {
        if (cur_line.length() == 0) {
          tcsetattr(STDIN_FILENO, TCSANOW, &old_settings);
          return 0;
        }

        eaten += key_seqs[MY_KEY_EOF].length();
      }
      else if (key_seqs[MY_KEY_LEFT].length() > 0 &&
               static_cast<size_t>(count - eaten) >= key_seqs[MY_KEY_LEFT].length() &&
               strncmp(&keycodes[eaten], key_seqs[MY_KEY_LEFT].c_str(), key_seqs[MY_KEY_LEFT].length()) == 0) {
        if (cur_pos > 0) {
          int cur_col = (cur_pos + 1 + strlen(PROMPT_STRING) + 1) % num_cols;
          if (cur_col == 0)
            cur_col = num_cols;          

          cur_pos--;

          move_cursor(cur_col, -1, num_cols, key_seqs);
        }
        eaten += key_seqs[MY_KEY_LEFT].length();
      }
      else if (key_seqs[MY_KEY_RIGHT].length() > 0 &&
               static_cast<size_t>(count - eaten) >= key_seqs[MY_KEY_RIGHT].length() &&
               strncmp(&keycodes[eaten], key_seqs[MY_KEY_RIGHT].c_str(), key_seqs[MY_KEY_RIGHT].length()) == 0) {
        if (static_cast<size_t>(cur_pos) < cur_line.length()) {
          int cur_col = (cur_pos + 1 + strlen(PROMPT_STRING) + 1) % num_cols;
          if (cur_col == 0)
            cur_col = num_cols;          
          
          cur_pos++;

          move_cursor(cur_col, 1, num_cols, key_seqs);
        }
        eaten += key_seqs[MY_KEY_RIGHT].length();
      }
      else if (key_seqs[MY_KEY_UP].length() > 0 &&
               static_cast<size_t>(count - eaten) >= key_seqs[MY_KEY_UP].length() &&
               strncmp(&keycodes[eaten], key_seqs[MY_KEY_UP].c_str(), key_seqs[MY_KEY_UP].length()) == 0) {
        if (cur_history_pos >= 0 && cur_history_pos != first_history_pos) {
          history[cur_history_pos] = cur_line;
          cur_history_pos = omega::int_mod(cur_history_pos-1, HISTORY_SIZE);

          int cur_col = (cur_pos + 1 + strlen(PROMPT_STRING) + 1) % num_cols;
          if (cur_col == 0)
            cur_col =  num_cols;

          move_cursor(cur_col, -cur_pos, num_cols, key_seqs);

          std::cout.write(history[cur_history_pos].c_str(), history[cur_history_pos].length());

          cur_col = (history[cur_history_pos].length() + strlen(PROMPT_STRING) + 1) % num_cols;
          if (cur_col == 0) {
            std::cout.put(' ');
            std::cout.put('\b');            
          }
          
          if (cur_line.length() > history[cur_history_pos].length()) {
            for (size_t i = 0; i < cur_line.length() - history[cur_history_pos].length(); i++)
              std::cout.put(' ');
            
            cur_col = (cur_line.length() + strlen(PROMPT_STRING) + 1) % num_cols;
            if (cur_col == 0)
              cur_col = num_cols + 1;
            else
              cur_col++;

            move_cursor(cur_col, -(cur_line.length() - history[cur_history_pos].length()), num_cols, key_seqs);
          }
          cur_line = history[cur_history_pos];
          cur_pos = cur_line.length();
        }
          
        eaten += key_seqs[MY_KEY_UP].length();
      }
      else if (key_seqs[MY_KEY_DOWN].length() > 0 &&
               static_cast<size_t>(count - eaten) >= key_seqs[MY_KEY_DOWN].length() &&
               strncmp(&keycodes[eaten], key_seqs[MY_KEY_DOWN].c_str(), key_seqs[MY_KEY_DOWN].length()) == 0) {
        if (cur_history_pos >= 0 && cur_history_pos != (last_history_pos+1)%HISTORY_SIZE) {
          history[cur_history_pos] = cur_line;
          cur_history_pos = (cur_history_pos+1)%HISTORY_SIZE;

          int cur_col = (cur_pos + 1 + strlen(PROMPT_STRING) + 1) % num_cols;
          if (cur_col == 0)
            cur_col =  num_cols;

          move_cursor(cur_col, -cur_pos, num_cols, key_seqs);
          
          std::cout.write(history[cur_history_pos].c_str(), history[cur_history_pos].length());

          cur_col = (history[cur_history_pos].length() + strlen(PROMPT_STRING) + 1) % num_cols;
          if (cur_col == 0) {
            std::cout.put(' ');
            std::cout.put('\b');
          }
          
          if (cur_line.length() > history[cur_history_pos].length()) {
            for (size_t i = 0; i < cur_line.length() - history[cur_history_pos].length(); i++)
              std::cout.put(' ');

            cur_col = (cur_line.length() + strlen(PROMPT_STRING) + 1) % num_cols;
            if (cur_col == 0)
              cur_col = num_cols + 1;
            else
              cur_col++;

            move_cursor(cur_col, -(cur_line.length() - history[cur_history_pos].length()), num_cols, key_seqs);
          }
          cur_line = history[cur_history_pos];
          cur_pos = cur_line.length();
        }
        
        eaten += key_seqs[MY_KEY_DOWN].length();
      } 
      else if (key_seqs[MY_KEY_DEL].length() > 0 &&
               static_cast<size_t>(count - eaten) >= key_seqs[MY_KEY_DEL].length() &&
               strncmp(&keycodes[eaten], key_seqs[MY_KEY_DEL].c_str(), key_seqs[MY_KEY_DEL].length()) == 0) {
        if (static_cast<size_t>(cur_pos) < cur_line.length()) {
          cur_line.erase(cur_pos, 1);
          std::cout.write(&(cur_line.c_str()[cur_pos]), cur_line.length()-cur_pos);
          std::cout.put(' ');

          int cur_col = (cur_line.length() + 1 + strlen(PROMPT_STRING) + 1) % num_cols;
          if (cur_col == 0)
            cur_col = num_cols + 1;
          else
            cur_col++;

          move_cursor(cur_col, -(cur_line.length()-cur_pos+1), num_cols, key_seqs);
        }
          
        eaten += key_seqs[MY_KEY_DEL].length();
      }
      else if (key_seqs[MY_KEY_HOME].length() > 0 &&
               static_cast<size_t>(count - eaten) >= key_seqs[MY_KEY_HOME].length() &&
               strncmp(&keycodes[eaten], key_seqs[MY_KEY_HOME].c_str(), key_seqs[MY_KEY_HOME].length()) == 0) {
        int cur_col = (cur_pos + 1 + strlen(PROMPT_STRING) + 1) % num_cols;
        if (cur_col == 0)
          cur_col = num_cols;

        move_cursor(cur_col, -cur_pos, num_cols, key_seqs);
        
        cur_pos = 0;
        eaten += key_seqs[MY_KEY_HOME].length();
      }
      else if (key_seqs[MY_KEY_END].length() > 0 &&
               static_cast<size_t>(count - eaten) >= key_seqs[MY_KEY_END].length() &&
               strncmp(&keycodes[eaten], key_seqs[MY_KEY_END].c_str(), key_seqs[MY_KEY_END].length()) == 0) {
        int cur_col = (cur_pos + 1 + strlen(PROMPT_STRING) + 1) % num_cols;
        if (cur_col == 0)
          cur_col = num_cols;

        move_cursor(cur_col, cur_line.length()-cur_pos, num_cols, key_seqs);

        cur_pos = cur_line.length();
        eaten += key_seqs[MY_KEY_END].length();
      }
      else if (key_seqs[MY_KEY_PGUP].length() > 0 &&
               static_cast<size_t>(count - eaten) >= key_seqs[MY_KEY_PGUP].length() &&
               strncmp(&keycodes[eaten], key_seqs[MY_KEY_PGUP].c_str(), key_seqs[MY_KEY_PGUP].length()) == 0) {
        eaten += key_seqs[MY_KEY_PGUP].length();
      }
      else if (key_seqs[MY_KEY_PGDOWN].length() > 0 &&
               static_cast<size_t>(count - eaten) >= key_seqs[MY_KEY_PGDOWN].length() &&
               strncmp(&keycodes[eaten], key_seqs[MY_KEY_PGDOWN].c_str(), key_seqs[MY_KEY_PGDOWN].length()) == 0) {
        eaten += key_seqs[MY_KEY_PGDOWN].length();
      }
      else if (key_seqs[MY_KEY_NUMPAD_HOME].length() > 0 &&
               static_cast<size_t>(count - eaten) >= key_seqs[MY_KEY_NUMPAD_HOME].length() &&
               strncmp(&keycodes[eaten], key_seqs[MY_KEY_NUMPAD_HOME].c_str(), key_seqs[MY_KEY_NUMPAD_HOME].length()) == 0) {
        eaten += key_seqs[MY_KEY_NUMPAD_HOME].length();
      }
      else if (key_seqs[MY_KEY_NUMPAD_END].length() > 0 &&
               static_cast<size_t>(count - eaten) >= key_seqs[MY_KEY_NUMPAD_END].length() &&
               strncmp(&keycodes[eaten], key_seqs[MY_KEY_NUMPAD_END].c_str(), key_seqs[MY_KEY_NUMPAD_END].length()) == 0) {
        eaten += key_seqs[MY_KEY_NUMPAD_END].length();
      }
      else if (keycodes[eaten] == '\x1B' && (count - eaten == 1 || keycodes[eaten+1] == '\x1B')) { // single ESC key
        eaten++;
      }
      else if (keycodes[eaten] == '\x1B') { // unknown escape sequences
        while (eaten+1 < count && keycodes[eaten+1] != '\x1B')
          eaten++;
        
        keycodes[eaten] = '~';
      }
      else if (keycodes[eaten] == '\x7F') { // backspace key
        if (cur_pos > 0) {
          int cur_col = (cur_pos + 1 + strlen(PROMPT_STRING) + 1) % num_cols;
          if (cur_col == 0)
            cur_col = num_cols;

          cur_pos--;
          cur_line.erase(cur_pos, 1);

          move_cursor(cur_col, -1, num_cols, key_seqs);
          
          std::cout.write(&(cur_line.c_str()[cur_pos]), cur_line.length()-cur_pos);
          std::cout.put(' ');
          
          cur_col = (cur_line.length() + 1 + strlen(PROMPT_STRING) + 1) % num_cols;
          if (cur_col == 0)
            cur_col = num_cols + 1;
          else
            cur_col++;

          move_cursor(cur_col, -(cur_line.length()-cur_pos+1), num_cols, key_seqs);
        }
            
        eaten++;
      }
      else if (keycodes[eaten] == '\n'){ // return key
        int cur_col = (cur_pos + 1 + strlen(PROMPT_STRING) + 1) % num_cols;
        if (cur_col == 0)
          cur_col = num_cols;

        move_cursor(cur_col, cur_line.length()-cur_pos, num_cols, key_seqs);
        
        std::cout.put(keycodes[eaten]);
        cur_line.append(1, '\n');
        cur_pos = 0;
        break;
      }
      else { // all other key
        std::cout.put(keycodes[eaten]);
        std::cout.write(&(cur_line.c_str()[cur_pos]), cur_line.length()-cur_pos);

        cur_line.insert(cur_pos, &keycodes[eaten], 1);
        cur_pos++;

        int cur_col = (cur_line.length() + strlen(PROMPT_STRING) + 1) % num_cols;
        if (cur_col == 0) {
          // force cursor to move to the next line when the last printed char is at
          // the right boundary of the terminal
          std::cout.put(' ');
          std::cout.put('\b');
          
          cur_col = 1;
        }
        else
          cur_col++;

        move_cursor(cur_col, -(cur_line.length()-cur_pos), num_cols, key_seqs);

        eaten++;
      }
        
      std::cout.flush();
    }
  }
#else
  return yyFlexLexer::LexerInput(buf, max_size);
#endif
}
