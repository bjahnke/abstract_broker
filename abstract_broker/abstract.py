import json
from abc import ABC, abstractmethod
import typing as t
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import reduce

from dataclasses import dataclass
from multiprocessing.connection import Connection

import pandas as pd
import multiprocessing as mp
import numpy as np


# ----------------------
# UTIL FUNCTIONS (START)
# ----------------------


class Side(Enum):
    """Used to express signals and trade direction in natural language"""

    LONG = 1
    SHORT = -1
    CLOSE = 0

    @classmethod
    def _missing_(cls, value):
        """
        provides additional mappings of enum values
        """
        enum_map = {
            "BUY": cls.LONG,
            "SELL_SHORT": cls.SHORT,
        }
        if (res := enum_map.get(value, None)) is None:
            if np.isnan(value):
                res = cls.CLOSE
            else:
                res = super()._missing_(value)
        return res


def set_bar_end_time(interval, time_stamp):
    time_remaining = interval - time_stamp.minute % interval
    # print(time_stamp.minute + time_remaining)
    minute_bound = time_stamp.minute + time_remaining
    hour_bound = time_stamp.hour
    if minute_bound == 60:
        minute_bound = 0
        hour_bound += 1
    right_bound_time = time_stamp.replace(
        hour=hour_bound, minute=minute_bound, second=0, microsecond=0
    )
    return right_bound_time


def get_target_fetch_time(stream_init_time, interval, data_delay=None):
    """
    given stream start time and interval of data,
    get the time when price history should be retreived
    to ensure no data gaps
    """
    time_remainder = timedelta(minutes=stream_init_time.minute % interval)
    if interval == 1:
        time_remainder += timedelta(minutes=1)

    target_fetch_time = stream_init_time.replace(
        second=0, microsecond=0
    ) + time_remainder

    if data_delay is not None:
        target_fetch_time += data_delay
    return target_fetch_time


# ----------------------
# UTIL FUNCTIONS (END)
# ----------------------


@dataclass
class Condition:
    case: t.Callable[[t.Any], bool]
    result: t.Any


Condition(case=lambda x: x > 0, result="buy")

Condition(case=lambda x: x < 0, result="buy")


class AbstractPosition(ABC):
    _raw_position: t.Dict
    _symbol: str
    _qty: float
    _side: Side
    _stop_value = float

    def __init__(
            self, symbol, qty, side, raw_position=None, stop_value=None, data_row=None
    ):
        self._symbol = symbol
        self._qty = qty
        self._side = Side(side)
        self._raw_position = raw_position
        self._stop_value = stop_value
        self._data_row = data_row
        self._stop_type = None

    @property
    def side(self):
        return self._side

    @property
    def qty(self):
        return self._qty

    def set_size(self, new_qty: int):
        """
        if new_qty != current quantity,
        execute an order for this position to attain
        the desired size
        :return:
        """
        assert new_qty >= 0, "input quantity must be >= 0"
        order_spec = None
        size_delta = new_qty - self._qty
        if size_delta != 0:
            self._qty = new_qty
            trade_qty = abs(size_delta)
            if size_delta > 0:
                """increase position size"""
                order_spec = self._open(quantity=trade_qty)
            else:
                """decrease position size"""
                order_spec = self._close(quantity=trade_qty)
        return order_spec

    def init_stop_loss(self, stop_type) -> t.Union[t.Callable]:
        order = None
        if self._qty > 0:
            self._stop_type = stop_type
            order = self._stop_order()
        return order

    @abstractmethod
    def _stop_order(self) -> t.Callable:
        raise NotImplementedError

    @abstractmethod
    def _open(self, quantity) -> t.Union[t.Callable, None]:
        raise NotImplementedError

    @abstractmethod
    def _close(self, quantity) -> t.Union[t.Callable, None]:
        raise NotImplementedError

    def open_order(self) -> t.Union[t.Callable, None]:
        return self._open(self.qty) if self._qty > 0 else None

    def full_close(self):
        """fully close the position"""
        self._close(quantity=self._qty)


class AbstractBrokerAccount(ABC):
    """get account info"""

    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def positions(self) -> t.Dict[str, t.Type[AbstractPosition]]:
        """get position info for all active positions"""
        raise NotImplementedError

    @property
    @abstractmethod
    def equity(self):
        raise NotImplementedError

    @abstractmethod
    def get_symbols(self) -> t.List[str]:
        """get all symbols of active positions within this account"""
        raise NotImplementedError


class ClientAlreadyExistsError(Exception):
    """
    a broker client of this type has already been initialized,
    cannot create another.
    """


class AbstractBrokerClient(ABC):
    """top level api client"""
    __instance_exists = False

    def __init__(self, credentials, *args, **kwargs):
        if self.__class__.__instance_exists is True:
            raise ClientAlreadyExistsError
        else:
            self.__class__.__instance_exists = True

        self._client = self.__class__._get_broker_client(credentials)

    @staticmethod
    @abstractmethod
    def _get_broker_client(credentials) -> t.Any:
        """
        Initializes the wrapped broker client object and returns object.
        AbstractBrokerClient uses this method to set its self._client
        attribute
        :param credentials: api access credentials for initializing the broker client
        :return: the broker client object
        """
        raise NotImplementedError

    @abstractmethod
    def account_info(self, *args, **kwargs):
        """
        Get account info of the broker this interface is wrapping
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def price_history(self, symbol, interval, interval_type, **kwargs) -> pd.DataFrame:
        """
        Get price history for a single symbol by the given date range and frequency
        :param symbol: equity symbol to get price history for
        :param freq_range: time range and interval of the price data to retrieve
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def place_order_spec(self, order_spec):
        """
        Sends an order object to the broker to be executed
        :param order_spec: contains specification details of the order
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_order_data(self, order_id, *args):
        """
        Retrieve the details of an order by the given order_id
        :param order_id:
        :param args:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def init_position(self, symbol, quantity, side):
        raise NotImplementedError

    @abstractmethod
    def init_stream(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def client(self):
        return self._client

    def download_price_history(self, symbols: t.List[str], interval, interval_type, **kwargs) -> pd.DataFrame:
        """
        Get price data for multiple signals.
        :param symbols:
        :param interval:
        :param interval_type:
        :param args:
        :param kwargs:
        :return:
        """
        dfs = []
        for i, symbol in enumerate(symbols):
            print(symbol)
            price_data = self.price_history(
                symbol=symbol, interval=interval, interval_type=interval_type, **kwargs
            )
            price_data.columns = pd.MultiIndex.from_product([[symbol], price_data.columns])
            dfs.append(price_data)

        dfs_merged = reduce(lambda left, right: left.join(right, how='outer'), dfs)
        return dfs_merged


class AbstractOrders(ABC):
    @abstractmethod
    def _long_open(self, ):
        pass

    @abstractmethod
    def _long_close(self):
        pass

    @abstractmethod
    def _short_open(self):
        pass

    @abstractmethod
    def _short_close(self):
        pass


class ReSize(Enum):
    INC = auto()
    DEC = auto()


OHLC_VALUES = t.Tuple[float, float, float, float]
DATA_FETCH_FUNCTION = t.Callable[[str, int], pd.DataFrame]


class CsvPermissionError(Exception):
    """PermissionError raised when attempting to read/write from price history csv"""


class LiveQuotePermissionError(Exception):
    """PermissionError raised when attempting to read/write from price history csv"""


class AbstractStreamParser(ABC):
    """
    class for defining how to parse stream data,
    additionally includes a state machine for
    """

    def __init__(self, interval, time_stamp, symbol):
        self._symbol = symbol
        self._interval = interval
        self._bar_end_time = set_bar_end_time(interval, time_stamp)
        self._open = None
        self._high = None
        self._low = None
        self._close = None

    def init_new(self, o, h, l, c):
        self._open = o
        self._high = h
        self._low = l
        self._close = c

    def update(self, o, h, l, c):
        try:
            self._low = min(self._low, l)
            self._high = max(self._high, h)
            self._close = c
        except TypeError:
            self.init_new(o, h, l, c)

    def get_ohlc_values(self) -> OHLC_VALUES:
        return self._open, self._high, self._low, self._close

    def update_ohlc(self, data: t.Dict, time_stamp) -> t.Dict[str, float]:
        values = self.retrieve_ohlc(data)
        if time_stamp > self._bar_end_time:
            self._bar_end_time = set_bar_end_time(self._interval, time_stamp)
            self.init_new(*values)
        else:
            self.update(*values)
        return self.get_ohlc()

    def get_ohlc(self):
        return {
            "open": self._open,
            "high": self._high,
            "low": self._low,
            "close": self._close,
            "symbol": self._symbol
        }

    @abstractmethod
    def retrieve_ohlc(self, data: dict) -> OHLC_VALUES:
        """get prices from ticker stream"""
        raise NotImplementedError


class AbstractTickerStream:
    """
    Outputs streamed data in OHLC format to csv file
    in the given interval
    """
    _stream_parser_cls: t.Type[AbstractStreamParser]
    _stream_parsers: t.Dict[str, AbstractStreamParser]
    _price_data: t.Union[pd.DataFrame]

    def __init__(
            self,
            stream_parser: t.Type[AbstractStreamParser],
            quote_file_path: str,
            history_path: str,
            fetch_price_data: DATA_FETCH_FUNCTION,
            interval: int = 1,
    ):
        """
        NOTE: fetch_price_data CANNOT be pickled if it is an instance attribute
        :param stream_parser:
        :param quote_file_path: path to live quote file, SHOULD NOT INCLUDE FILE NAME
        :param history_path: path to output history file, SHOULD NOT INCLUDE FILE NAME
        :param fetch_price_data:
        :param interval:
        """
        # self.__class__._validate_path(
        #     (quote_file_path, '.json'),
        #     (history_path, '.csv')
        # )
        self._stream_parser_cls = stream_parser
        self._quote_file_path = quote_file_path
        self._history_path = history_path
        self._interval = interval
        self._stream_parsers = {}
        self._fetch_price_data = fetch_price_data
        self._columns = ["symbol", "open", "high", "low", "close"]

    @staticmethod
    def _validate_path(*path_check: t.Tuple[str, str]):
        """ensures each file path contains the expected file extension"""
        for file_path, file_type_check in path_check:
            assert file_type_check in file_path[-len(file_type_check):]

    @abstractmethod
    def run_stream(self, *args, **kwargs):
        raise NotImplementedError

    def get_fetch_time(self, stream_start_time, data_delay):
        return get_target_fetch_time(stream_start_time, self._interval, data_delay)

    @staticmethod
    @abstractmethod
    def get_symbol(msg) -> str:
        raise NotImplementedError

    def _init_processes(self, writer_send_conn):
        """
        Initialize a separate process for writing data to file
        :param:
        :return:
        """
        receive_conn, send_conn = mp.Pipe(duplex=False)
        write_process = mp.Process(
            target=_write_row_handler,
            args=(
                self._interval,
                self._columns,
                self._history_path,
                receive_conn,
                writer_send_conn,
            )
        )
        write_process.start()

        return send_conn

    # def handle_stream(
    #     self, current_quotes, queue: mp.SimpleQueue, send_conn: Connection
    # ):
    #     """handles the messages, translates to ohlc values, outputs to json and csv"""
    #     # start_time = time()
    #     while True:
    #         msg = queue.get()
    #         symbol = self.__class__.get_symbol(msg)
    #         self._stream_parsers[symbol].update_ohlc_state(msg)
    #         ohlc_data = self._stream_parsers[symbol].get_ohlc()
    #         current_quotes[symbol] = ohlc_data
    #         send_conn.send(current_quotes)

    def _init_stream_parsers(self, symbols, stream_start_time):
        """
        initialize stream parser for all symbols in stream,
        where additional delay relates to time delay of price history data to account for,
        and a dict of params to be passed into the fetch price history function
        :param args:
        :return:
        """
        self._stream_parsers = {
            symbol: (
                self._stream_parser_cls(
                    symbol=symbol,
                    time_stamp=stream_start_time,
                    interval=self._interval,
                )
            )
            for symbol in symbols
        }

    def get_all_symbol_data(self, symbols, interval) -> pd.DataFrame:
        """get all price history and return concat dataframe of all data"""
        all_data = []
        for symbol in symbols:
            data = self._fetch_price_data(symbol, interval)
            data['symbol'] = symbol
            all_data.append(data)
        return pd.concat(all_data)


def _write_row_handler(
        interval,
        columns,
        history_path,
        receive_conn: Connection,
        send_conn: Connection
):
    """
    NOTE: Pure function for compatability with multiprocess
    wait until until the current bar time is exceeded, then write
    the current content of the receive connection as a new row
    :param receive_conn:
    :return:
    """

    # TODO PRINT lag
    bar_end_time = set_bar_end_time(interval, datetime.utcnow())
    current_quotes = None

    while True:
        time_stamp = datetime.utcnow()
        if receive_conn.poll():
            current_quotes = receive_conn.recv()
        if time_stamp > bar_end_time:
            pre_write_lag = time_stamp - bar_end_time
            if current_quotes is None:
                # don't do anything until we receive the first message
                continue

            symbols_written = []
            if None in current_quotes.values():
                idx = list(current_quotes.values()).index(None)
                raise Exception(f'Not receiving data from {list(current_quotes.keys())[idx]}')

            price_datas = [price_data for price_data in current_quotes.values() if price_data is not None]
            new_price_data = pd.DataFrame(price_datas, index=[bar_end_time] * len(price_datas))
            new_price_data.to_csv(history_path, mode='a', header=False)
            post_write_lag = datetime.utcnow() - bar_end_time
            print(
                f"{bar_end_time} "
                f"(pre-write lag): {pre_write_lag}, "
                f"(post-write lag): {post_write_lag}"
            )

            send_conn.send(True)

            # shift bar end time to the right by 1 interval
            bar_end_time = set_bar_end_time(interval, time_stamp)
