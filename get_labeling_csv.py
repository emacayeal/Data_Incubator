from util.sa_func import get_label_df
import pandas as pd
import yaml
import os
import pickle as pkl


def get_additional_cols(df, add_cols):
    add_list = [[] for _ in add_cols]
    for idx, ac in enumerate(add_cols):
        if ac == 'rating':
            try:
                add_list[idx].extend(df['rating'].tolist())
            except KeyError:
                try:
                    add_list[idx].extend(df['rating_x'].tolist())
                except KeyError:
                    raise Exception('no rating column present')
        else:
            add_list[idx].extend(df[ac].tolist())
    return add_list


config_file = './configs/get_label_config.yaml'

with open(config_file, 'r') as yml_file:
    cfg = yaml.load(yml_file)

# read in pairs of csv's with reviews and product categories
rev_cat_pairs = cfg['data_paths']['review_category_pairs']
label_out_dir = cfg['data_paths']['label_out_dir']
if not os.path.exists(label_out_dir):
    os.makedirs(label_out_dir)

# categories to subset on, attributes to label
use_cats = cfg['categories']['use_cols']
attr_list = cfg['attributes']
add_cols = cfg['add_cols']
use_vader = cfg['sentiment']['use_vader']
use_tb = cfg['sentiment']['use_tb']

# number of review samples
num_samples = cfg['num_samples']
min_sentence_length = cfg['min_sentence_length']

# merge all of the df pairs
df_list = []
print('merging df pairs')
for pair in rev_cat_pairs:
    rdf = pd.read_csv(pair[0])
    cdf = pd.read_csv(pair[1])
    # merge and append
    rdf = pd.merge(rdf, cdf, how='inner', on='productId')
    df_list.append(rdf)

# get review and ratings list for different categories
for cat in use_cats:
    print('working on %s category' % cat)
    rev_list = []
    add_list = [[] for _ in add_cols] + [[]]
    for df in df_list:
        if cat == 'all':
            sdf = df
        else:
            sdf = df[df[cat] == 1]
        rev_list.extend([str(x).replace('\n', ' ') for x in sdf['reviewText'].tolist()])
        tmp_add_list = get_additional_cols(sdf, add_cols + [cat + '_desc'])
        for i in range(len(add_list)):
            add_list[i].extend(tmp_add_list[i])
    # pkl.dump(rev_list, open(label_out_dir + cat + '_rev_list.pkl', 'wb'))
    # now get the sentiments and create labeling csv's
    labeling_df = get_label_df(rev_list, attr_list, num_samples, add_list,
                               add_cols + [cat + '_desc'], min_sentence_length, use_vader, use_tb)
    out_str = cat + '_labeling.csv'
    labeling_df.to_csv(label_out_dir + out_str, index=False)