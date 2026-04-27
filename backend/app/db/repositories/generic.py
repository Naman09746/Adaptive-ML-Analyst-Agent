# ama2/backend/app/db/repositories/generic.py
from typing import Generic, TypeVar, Type, Optional, List
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ..base import Base

T = TypeVar("T", bound=Base)

class GenericRepository(Generic[T]):
    def __init__(self, model: Type[T], session: AsyncSession):
        self.model = model
        self.session = session

    async def get_by_id(self, id: UUID) -> Optional[T]:
        return await self.session.get(self.model, id)

    async def create(self, **kwargs) -> T:
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def filter(self, **kwargs) -> List[T]:
        stmt = select(self.model).filter_by(**kwargs)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def delete(self, id: UUID) -> bool:
        instance = await self.get_by_id(id)
        if instance:
            await self.session.delete(instance)
            return True
        return False
