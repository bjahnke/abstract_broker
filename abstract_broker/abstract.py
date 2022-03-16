import json
from abc import ABC, abstractmethod
import typing as t
from datetime import datetime, timedelta
from enum import Enum, auto

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
    right_bound_time = time_stamp.replace(
        minute=time_stamp.minute + time_remaining, second=0, microsecond=0
    )
    return right_bound_time


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
    __instance_exists = False
    """top level api client"""
    def __init__(self, credentials, *args, **kwargs):
        if self.__class__.__instance_exists is True:
            raise ClientAlreadyExistsError
        else:
            self.__class__.__instance_exists = True
            
        self._client = self.__class__._get_broker_client(credentials)

    @staticmethod
    @abstractmethod
    def _get_broker_client(credentials) -> t.Any:
        raise NotImplementedError

    @abstractmethod
    def account_info(self, *args, **kwargs) -> t.Type[AbstractBrokerAccount]:
        raise NotImplementedError

    @abstractmethod
    def price_history(self, symbol, freq_range):
        raise NotImplementedError

    @abstractmethod
    def place_order_spec(self, order_spec):
        raise NotImplementedError

    @abstractmethod
    def get_order_data(self, order_id, *args):
        raise NotImplementedError

    @abstractmethod
    def init_position(self, symbol, quantity, side):
        raise NotImplementedError


class AbstractOrders(ABC):
    @abstractmethod
    def _long_open(self,):
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
DATA_FETCH_FUNCTION = t.Callable[[str, int, int, str], t.Tuple[pd.DataFrame, t.Any]]


class CsvPermissionError(Exception):
    """PermissionError raised when attempting to read/write from price history csv"""


class LiveQuotePermissionError(Exception):
    """PermissionError raised when attempting to read/write from price history csv"""


class Bar:
    def __init__(self, interval, time_stamp):
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

    def update_ohlc(self, values: OHLC_VALUES, time_stamp):
        if time_stamp > self._bar_end_time:
            self._bar_end_time = set_bar_end_time(self._interval, time_stamp)
            self.init_new(*values)
        else:
            self.update(*values)

    def get_ohlc(self):
        return {
            "open": self._open,
            "high": self._high,
            "low": self._low,
            "close": self._close,
        }


class StreamState(Enum):
    INITIAL = auto()
    WAIT = auto()
    FILL_GAP = auto()
    NORMAL_UPDATE = auto()
    INIT_NEW_BAR = auto()
    ADD_NEW_BAR = auto()


class AbstractStreamParser(ABC):
    """
    class for defining how to parse stream data,
    additionally includes a state machine for
    """
    _quoted_prices: OHLC_VALUES
    _fetch_price_data: DATA_FETCH_FUNCTION
    _state_table: t.Dict[StreamState, t.Callable[[datetime.timestamp], StreamState]]

    def __init__(
        self,
        symbol,
        fetch_price_data: DATA_FETCH_FUNCTION,
        data_delay,
        live_quote_file_path=None,
        history_path="",
        interval: int = 1,
    ):
        self._data_delay = data_delay
        self._stream_state = StreamState.INITIAL
        self._stream_init_time = None
        self._target_fetch_time = None
        self._live_quote_file_path = live_quote_file_path
        self._symbol = symbol
        self._fetch_price_data = fetch_price_data
        self._history_file_path = self.__class__._init_price_history_path(
            history_path, symbol
        )
        self._interval = interval
        self._bar_data = None
        self._quoted_prices = (0, 0, 0, 0)
        self._prev_sequence = None

        self._state_table = {
            StreamState.INITIAL: self._init_stream_data,
            StreamState.FILL_GAP: self._allow_fill_data_gap,
            StreamState.NORMAL_UPDATE: self._do_nothing,
        }

    @abstractmethod
    def retrieve_ohlc(self, data: dict) -> OHLC_VALUES:
        """get prices from ticker stream"""
        raise NotImplementedError

    @property
    def history_file_path(self):
        return

    def fetch_price_data(self) -> t.Tuple[pd.DataFrame, t.Any]:
        """x days worth of minute data by the given interval"""
        x = 5
        return self._fetch_price_data(self._symbol, self._interval, x, "m")

    def update_ohlc_state(self, data: t.Dict):
        """"""
        self._quoted_prices = self.retrieve_ohlc(data)
        time_stamp = datetime.utcnow()

        self._stream_state = self._state_table[self._stream_state](time_stamp)

        self._bar_data.update_ohlc(self._quoted_prices, time_stamp)

    def _init_stream_data(self, time_stamp) -> StreamState:
        """
        make an initial price history call,
        calculate the amount of time needed for the
        stream to run to close the data gap between
        price history end and stream start
        """
        self._stream_init_time = time_stamp
        self._bar_data = Bar(self._interval, time_stamp)
        next_bar_time = self._stream_init_time.replace(
            second=0, microsecond=0
        ) + timedelta(minutes=self._stream_init_time.minute % self._interval)
        self._target_fetch_time = next_bar_time + self._data_delay
        return StreamState.FILL_GAP

    def _allow_fill_data_gap(self, time_stamp) -> StreamState:
        """
        wait for current time to exceed target time,
        get price history again. Now there is no longer a data gap
        """
        next_state = StreamState.FILL_GAP
        if time_stamp > self._target_fetch_time:
            price_data, _ = self.fetch_price_data()
            price_data.to_csv(self._history_file_path)
            print(f"{self._symbol} gap filled")
            next_state = StreamState.NORMAL_UPDATE
        return next_state

    def _do_nothing(self, _):
        return self._stream_state

    def get_ohlc(self) -> t.Dict:
        res = self._bar_data.get_ohlc()
        res["symbol"] = self._symbol
        return res

    @staticmethod
    def _init_price_history_path(price_history_path, symbol):
        full_path = f"{symbol}.ftr"
        if len(price_history_path) > 0:
            full_path = f"{price_history_path}\\{full_path}"
        return full_path


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
        self._stream_parser_cls = stream_parser
        self._quote_file_path = quote_file_path
        self._history_path = history_path
        self._interval = interval
        self._stream_parsers = {}
        # self._live_data_out = {}
        self._fetch_price_data = fetch_price_data
        # self._permission_error_count = 0
        self._msg_queue_lookup = {}

        self._columns = ["symbol", "open", "high", "low", "close"]

    @abstractmethod
    def run_stream(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_symbol(msg) -> str:
        raise NotImplementedError

    def _init_processes(self):
        """
        Initialize a separate process for writing data to file
        :param:
        :return:
        """
        receive_conn, send_conn = mp.Pipe(duplex=False)
        write_process = mp.Process(target=self._write_row_handler, args=(receive_conn,))
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

    def _write_row_handler(self, receive_conn: Connection):
        """
        wait until until the current bar time is exceeded, then write
        the current content of the receive connection as a new row
        :param receive_conn:
        :return:
        """
        # TODO PRINT lag
        bar_end_time = set_bar_end_time(self._interval, datetime.utcnow())
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

                for symbol, price_data in current_quotes.items():
                    if price_data is not None:
                        new_row = pd.DataFrame(
                            [price_data.values()],
                            columns=self._columns,
                            index=[bar_end_time],
                        )
                        new_row.to_csv(
                            self.get_price_history_file_path(symbol),
                            mode="a",
                            header=False,
                        )
                        post_write_lag = datetime.utcnow() - bar_end_time
                        print(
                            f"{symbol} {bar_end_time} "
                            f"(pre-write lag): {pre_write_lag}, "
                            f"(post-write lag): {post_write_lag}"
                        )

                # shift bar end time to the right by 1 interval
                bar_end_time = set_bar_end_time(self._interval, time_stamp)

    def get_price_history_file_path(self, symbol: str):
        full_path = f"{symbol}.csv"
        if len(self._history_path) > 0:
            full_path = f"{self._history_path}\\{full_path}"
        return full_path

    def _init_stream_parsers(self, symbol_delays: t.Tuple[str, t.Any]):
        self._stream_parsers = {
            symbol: (
                self._stream_parser_cls(
                    symbol,
                    data_delay=delay,
                    interval=self._interval,
                    fetch_price_data=self._fetch_price_data,
                    history_path=self._history_path,
                )
            )
            for symbol, delay in symbol_delays
        }
