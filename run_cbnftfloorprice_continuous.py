import logging
import pandas as pd
import numpy as np
import cbnftfloorprice
from functools import partial


LOOKBACK = 140
BACKTEST = 800
PCT_TARGET = 0.05
PCT_TARGET_MIN = 0.02
PCT_TARGET_MAX = 0.1
SPEED = 0.5


def main() -> None:
    logging.info("reading data")
    filename = 'nft_trades_v2'
    nft_trades_df = pd.read_csv(f"{filename}.csv")
    # TODO: Make collection addr an input
    nft_trades_df = \
        nft_trades_df[
            nft_trades_df.contract_address ==\
                '0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D'.lower()
            ]
    logging.info("preprocessing")
    nft_trades_df = nft_trades_df[nft_trades_df["price_eth"] > 0]
    nft_trades_df["log_price"] = np.log(nft_trades_df["price_eth"])
    nft_trades_df.sort_values(
        ["collection", "block_number"], inplace=True
    )

    logging.info("creating lookback")
    nft_trades_df = (
        nft_trades_df.groupby(by=["collection"])
        .apply(partial(cbnftfloorprice.create_lookback, lookback=LOOKBACK))
        .reset_index(drop=True)
    )

    logging.info("removing outliers")
    nft_trades_df["log_prices_lookback_no_outliers"] = nft_trades_df.apply(
        lambda x: cbnftfloorprice.remove_outliers(x["log_prices_lookback"]),
        axis=1,
    )

    logging.info("compute target quantile")
    nft_trades_df["log_price_target_quantile"] = nft_trades_df.apply(
        lambda x: cbnftfloorprice.compute_quantile(
            x["log_prices_lookback_no_outliers"], PCT_TARGET
        ),
        axis=1,
    )

    logging.info("compute observed quantile")
    nft_trades_df["price_smaller"] = nft_trades_df.apply(
        lambda x: x["log_price"] <= x["log_price_target_quantile"],
        axis=1,
    )
    nft_trades_df["one"] = 1
    nft_trades_grouped_df = (
        nft_trades_df[["collection", "price_smaller", "one"]]
        .groupby(["collection"])
        .rolling(800)
        .sum()
    ).reset_index().drop('level_2', axis=1)
    nft_trades_grouped_df = nft_trades_grouped_df.dropna()
    nft_trades_grouped_df["quantile_obs"] = (
        nft_trades_grouped_df["price_smaller"] / nft_trades_grouped_df["one"]
    )

    logging.info("compute adjusted quantile")
    nft_trades_grouped_df["quantile_adj"] = nft_trades_grouped_df.apply(
        lambda x: cbnftfloorprice.compute_new_quantile(
            PCT_TARGET,
            PCT_TARGET,
            x["quantile_obs"],
            SPEED,
            PCT_TARGET_MIN,
            PCT_TARGET_MAX,
        ),
        axis=1,
    )

    logging.info("computing adjusted log price")
    nft_trades_df = nft_trades_df.drop(["price_smaller", "one"], axis=1,).join( 
        nft_trades_grouped_df[['quantile_adj']], how='inner'
    )
    nft_trades_df["log_price_adj"] = nft_trades_df.apply(
        lambda x: cbnftfloorprice.compute_quantile(
            x["log_prices_lookback_no_outliers"],
            x["quantile_adj"],
        ),
        axis=1,
    )

    nft_trades_df["floor_price_est"] = np.exp(nft_trades_df["log_price_adj"])
    nft_trades_df.to_csv(f'{filename}_results.csv')


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    main()
