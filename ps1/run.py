import argparse
from pathlib import Path

from .p01b_logreg import main as p01b
from .p01e_gda import main as p01e
from .p02cde_posonly import main as p02
from .p03d_poisson import main as p03
from .p05b_lwr import main as p05b
from .p05c_tau import main as p05c

from .p01fg_plot import main as p01fg, main_h as p01h

parser = argparse.ArgumentParser()
parser.add_argument('p_num', nargs='?', type=int, default=0,
                    help='Problem number to run, 0 for all problems.')
args = parser.parse_args()

CURR_DIR = Path(__file__).resolve().parent

# Problem 1
if args.p_num == 0 or args.p_num == 1:
    p01b(train_path=CURR_DIR / '../data/ds1_train.csv',
         eval_path=CURR_DIR / '../data/ds1_valid.csv',
         pred_path=CURR_DIR / '../output/p01b_pred_1.txt')

    p01b(train_path=CURR_DIR / '../data/ds2_train.csv',
         eval_path=CURR_DIR / '../data/ds2_valid.csv',
         pred_path=CURR_DIR / '../output/p01b_pred_2.txt')

    p01e(train_path=CURR_DIR / '../data/ds1_train.csv',
         eval_path=CURR_DIR / '../data/ds1_valid.csv',
         pred_path=CURR_DIR / '../output/p01e_pred_1.txt')

    p01e(train_path=CURR_DIR / '../data/ds2_train.csv',
         eval_path=CURR_DIR / '../data/ds2_valid.csv',
         pred_path=CURR_DIR / '../output/p01e_pred_2.txt')

if args.p_num == 7:
    p01fg(train_path=CURR_DIR / '../data/ds1_train.csv',
          title='Logistic Regression VS GDA Decision Boundaries (ds1_train)',
          save_path=CURR_DIR / '../output/p01f.png')
    p01fg(train_path=CURR_DIR / '../data/ds2_train.csv',
          title='Logistic Regression VS GDA Decision Boundaries (ds2_train)',
          save_path=CURR_DIR / '../output/p01g.png')

if args.p_num == 8:
    p01h(train_path=CURR_DIR / '../data/ds1_train.csv',
         train_title='Logistic Regression VS GDA Decision Boundaries (ds1_train) improved',
         save_train_path=CURR_DIR / '../output/p01h_train.png',
         valid_path=CURR_DIR / '../data/ds1_valid.csv',
         valid_title='Logistic Regression VS GDA Decision Boundaries (ds1_valid) improved',
         save_valid_path=CURR_DIR / '../output/p01h_valid.png')

# Problem 2
if args.p_num == 0 or args.p_num == 2:
    p02(train_path=CURR_DIR / '../data/ds3_train.csv',
        valid_path=CURR_DIR / '../data/ds3_valid.csv',
        test_path=CURR_DIR / '../data/ds3_test.csv',
        pred_path=CURR_DIR / '../output/p02X_pred.txt')

# Problem 3
if args.p_num == 0 or args.p_num == 3:
    p03(lr=1e-7,
        train_path=CURR_DIR / '../data/ds4_train.csv',
        eval_path=CURR_DIR / '../data/ds4_valid.csv',
        pred_path=CURR_DIR / '../output/p03d_pred.txt')

# Problem 5
if args.p_num == 0 or args.p_num == 5:
    p05b(tau=5e-1,
         train_path=CURR_DIR / '../data/ds5_train.csv',
         eval_path=CURR_DIR / '../data/ds5_valid.csv')

    p05c(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path=CURR_DIR / '../data/ds5_train.csv',
         valid_path=CURR_DIR / '../data/ds5_valid.csv',
         test_path=CURR_DIR / '../data/ds5_test.csv',
         pred_path=CURR_DIR / '../output/p05c_pred.txt')
