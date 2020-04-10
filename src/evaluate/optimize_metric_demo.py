"""
For both Hamming Space Retrieval (MAP@H<=2) and Ranking Retrieval (MAP@TopK),
we carefully optimize the measurement functions.
In usual, we speedup them by O(logK)/O(logN).

In this scripts, we provided some examples to demonstrate their efficiency.
"""
import argparse
import pickle

import numpy as np
import os
import sys
import time

sys.path.append('..')
sys.path.append(os.path.abspath('../valid'))

from evaluate.measure_utils \
    import get_precision_recall_by_Hamming_Radius, \
    get_precision_recall_by_Hamming_Radius_optimized, mean_average_precision_normal, \
    mean_average_precision_normal_optimized_label, mean_average_precision_normal_optimized_topK


if __name__ == "__main__":
    print("sync sync" + "*" * 20)
    print(os.getcwd())
    parser = argparse.ArgumentParser(description='check-result')
    parser.add_argument('--dir_name', type=str, default='../output/coco_48bit',
                        help="dir name")
    parser.add_argument('--format', type=str, default="default", help="default, ADSH")
    parser.add_argument('--test_radius', type=str, default="True", help="test_radius")
    parser.add_argument('--R', type=int, default=0, help="recall@R, 0 for no testing it")
    parser.add_argument('--verbose', type=int, default='1', help="verbose level")
    parser.add_argument('--sample_ratio', type=float, default=1, help="verbose level")
    print(os.getcwd())
    args = parser.parse_args()
    dir_name = args.dir_name
    R = args.R
    test_radius = args.test_radius == 'True'

    ## ========= special for local PyCharm test start =========
    # dir_name = '../../test_output/models_coco_48_mmhh_seen'
    # test_radius = True
    # R = 5000
    ## ========= special for local PyCharm test end =========

    print('valid file: %s' % dir_name)
    verbose = args.verbose
    if args.format == 'default':
        query_output = np.load(dir_name + '/test_code.npy')
        query_labels = np.load(dir_name + '/test_labels.npy')
        database_output = np.load(dir_name + '/database_code.npy')
        database_labels = np.load(dir_name + '/database_labels.npy')
    elif args.format == 'ADSH':
        query_output, query_labels, database_output, database_labels = load_ADSH(dir_name)
    else:
        raise NotImplementedError
    if args.sample_ratio < 1:
        query_len = int(query_output.shape[0] * args.sample_ratio)
        base_len = int(database_output.shape[0] * args.sample_ratio)
        # query_len = min(3, query_labels.shape[0])
        # base_len = min(5, database_labels.shape[0])
        # query_len = min(20, query_labels.shape[0])
        # base_len = min(100, database_labels.shape[0])
        query_output = query_output[:query_len]
        query_labels = query_labels[:query_len]
        database_output = database_output[:base_len]
        database_labels = database_labels[:base_len]
        print("sample %.2f%% query and base, query len: %d, base len: %d" %
              (args.sample_ratio * 100, query_len, base_len))
    # database_output = database_output[:50, :20]
    # database_labels = database_labels[:50, :20]
    # query_output = query_output[:30, :20]
    # query_labels = query_labels[:30, :20]

    output_dim = query_output.shape[1]
    if test_radius:
        line_prec = []
        line_rec = []
        line_mmap = []
        line_time = []
        # prec, rec, mmap = get_precision_recall_by_Hamming_Radius_All(img_database, img_query)
        # for i in range(output_dim + 1):
        #     print('Results ham dist [%d], prec:%s, rec:%s, mAP:%s' % (i, prec[i], rec[i], mmap[i]))

        print("test target radius")
        start = time.time()
        prec, rec, mmap = get_precision_recall_by_Hamming_Radius(
            database_output, database_labels, query_output, query_labels, 2)
        end = time.time()
        line_prec.append(prec)
        line_rec.append(rec)
        line_mmap.append(mmap)
        line_time.append(end - start)

        print("test radius refine")
        start = time.time()
        prec, rec, mmap, label_matchs = get_precision_recall_by_Hamming_Radius_optimized(
            database_output, database_labels, query_output, query_labels, 2)
        end = time.time()
        line_prec.append(prec)
        line_rec.append(rec)
        line_mmap.append(mmap)
        line_time.append(end - start)

        time.sleep(0.1)
        print("rate\titem\tstd  \trefine")
        print("rate %.2f\t" % args.sample_ratio + "prec\t" + "\t".join(["%.4f" % l for l in line_prec]))
        print("rate %.2f\t" % args.sample_ratio + "recall\t" + "\t".join(["%.4f" % l for l in line_rec]))
        print("rate %.2f\t" % args.sample_ratio + "mmap\t" + "\t".join(["%.4f" % l for l in line_mmap]))
        print("rate %.2f\t" % args.sample_ratio + "time\t" + "\t".join(["%.4f" % l for l in line_time]))

    print("\n")
    if R > 0:
        R = int(args.sample_ratio * R)
        line_prec = []
        line_rec = []
        line_mmap = []
        line_time = []

        print("test linear mAP")
        start = time.time()
        prec_norm, rec_norm, mmap_norm = mean_average_precision_normal(
            database_output, database_labels, query_output, query_labels, R)
        end = time.time()

        line_prec.append(prec_norm)
        line_rec.append(rec_norm)
        line_mmap.append(mmap_norm)
        line_time.append(end - start)

        print("test linear mAP refine label")
        start = time.time()
        prec_norm, rec_norm, mmap_norm = mean_average_precision_normal_optimized_label(
            database_output, database_labels, query_output, query_labels, R)
        end = time.time()
        print(end - start)
        line_prec.append(prec_norm)
        line_rec.append(rec_norm)
        line_mmap.append(mmap_norm)
        line_time.append(end - start)

        print("test linear mAP huge refine topK")
        start = time.time()
        prec_norm, rec_norm, mmap_norm = mean_average_precision_normal_optimized_topK(
            database_output, database_labels, query_output, query_labels, R)
        end = time.time()
        line_prec.append(prec_norm)
        line_rec.append(rec_norm)
        line_mmap.append(mmap_norm)
        line_time.append(end - start)

        time.sleep(0.1)

        print("rate item\tstd  \tfaster\tfastest")
        print("rate %.2f\t" % args.sample_ratio + "prec\t" + "\t".join(["%.4f" % l for l in line_prec]))
        print("rate %.2f\t" % args.sample_ratio + "recall\t" + "\t".join(["%.4f" % l for l in line_rec]))
        print("rate %.2f\t" % args.sample_ratio + "mmap\t" + "\t".join(["%.4f" % l for l in line_mmap]))
        print("rate %.2f\t" % args.sample_ratio + "time\t" + "\t".join(["%.4f" % l for l in line_time]))
