from aiohttp import ClientSession


class Binance:

    __base_url: str = "https://api.binance.com"

    def __init__(self, request_kwarg: dict = None) -> None:
        if not request_kwarg:
            request_kwarg = {}
        self.request_kwarg = request_kwarg

    async def __request(self, *args, **kwargs) -> dict:
        async with ClientSession(base_url=self.__base_url, **self.request_kwarg) as session:
            async with session.request(*args, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                raise Exception(response)

    async def __call_method(self, url, *args, **kwargs) -> dict:
        return await self.__request('get', f'/api/v3/{url}', *args, **kwargs)

    async def ping(self) -> dict:
        return await self.__call_method('ping')

    async def server_time(self) -> dict:
        return await self.__call_method('time')

    async def exchange_info(self) -> dict:
        return await self.__call_method('exchangeInfo')

    async def order_book(self, symbol: str, limit: int = 500) -> dict:
        return await self.__call_method('depth', params={
            'symbol': symbol,
            'limit': limit
        })

    async def trades_list(self, symbol: str, limit: int = 500) -> dict:
        return await self.__call_method('trades', params={
            'symbol': symbol,
            'limit': limit
        })

    async def trades_history(self, symbol: str, limit: int = 500, from_id: int = None) -> dict:
        return await self.__call_method('historicalTrades', params={
            'symbol': symbol,
            'limit': limit,
            'fromId': from_id
        })

    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> dict:
        return await self.__call_method('klines', params={
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        })

    async def price_symbol(self, symbol: str) -> dict:
        return await self.__call_method('ticker/price', params={
            'symbol': symbol
        })
