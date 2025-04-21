"""
Batch Processor Module
Provides concurrent and batched processing capabilities for large datasets.
"""
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, Generic

import numpy as np
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 10
    max_concurrency: int = 5
    timeout_per_batch: int = 60
    show_progress: bool = True
    retry_count: int = 3
    retry_delay: int = 5
    track_metrics: bool = True


class BatchProcessor(Generic[T, R]):
    """
    Batch processor for efficient processing of large data collections.
    
    Provides both synchronous and asynchronous processing capabilities
    with configurable batch sizes, concurrency levels, and timeouts.
    """
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Initialize the batch processor.
        
        Args:
            config: Configuration for the batch processor
        """
        self.config = config or BatchConfig()
        self.metrics = {
            "total_items": 0,
            "processed_items": 0,
            "successful_items": 0,
            "failed_items": 0,
            "total_time": 0,
            "batches_processed": 0,
            "retries": 0,
        }
    
    def process(
        self, 
        items: List[T], 
        process_func: Callable[[T], R],
        on_batch_complete: Optional[Callable[[List[R]], None]] = None
    ) -> List[R]:
        """
        Process items in batches using ThreadPoolExecutor for concurrency.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            on_batch_complete: Optional callback after each batch completes
            
        Returns:
            List of processed results
        """
        self._reset_metrics()
        self.metrics["total_items"] = len(items)
        start_time = time.time()
        
        # Prepare batches
        batches = self._create_batches(items)
        results = []
        
        # Set up progress bar if enabled
        pbar = None
        if self.config.show_progress:
            pbar = tqdm(total=len(items), desc="Processing items")
        
        # Process batches
        for i, batch in enumerate(batches):
            batch_results = self._process_batch(batch, process_func)
            results.extend(batch_results)
            
            # Update metrics
            self.metrics["batches_processed"] += 1
            self.metrics["processed_items"] += len(batch_results)
            
            # Call batch completion callback if provided
            if on_batch_complete:
                on_batch_complete(batch_results)
            
            # Update progress bar
            if pbar:
                pbar.update(len(batch))
        
        # Close progress bar
        if pbar:
            pbar.close()
        
        # Update final metrics
        self.metrics["total_time"] = time.time() - start_time
        
        return results
    
    async def process_async(
        self, 
        items: List[T], 
        process_func: Callable[[T], R],
        on_batch_complete: Optional[Callable[[List[R]], None]] = None
    ) -> List[R]:
        """
        Process items in batches asynchronously.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            on_batch_complete: Optional callback after each batch completes
            
        Returns:
            List of processed results
        """
        self._reset_metrics()
        self.metrics["total_items"] = len(items)
        start_time = time.time()
        
        # Prepare batches
        batches = self._create_batches(items)
        results = []
        
        # Set up progress bar if enabled
        pbar = None
        if self.config.show_progress:
            pbar = tqdm(total=len(items), desc="Processing items asynchronously")
        
        # Process batches concurrently
        batch_tasks = []
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        
        async def process_batch_with_semaphore(batch):
            async with semaphore:
                return await self._process_batch_async(batch, process_func)
        
        for batch in batches:
            batch_task = asyncio.create_task(process_batch_with_semaphore(batch))
            batch_tasks.append(batch_task)
        
        # Wait for all batch tasks to complete
        for i, task in enumerate(asyncio.as_completed(batch_tasks)):
            batch_results = await task
            results.extend(batch_results)
            
            # Update metrics
            self.metrics["batches_processed"] += 1
            self.metrics["processed_items"] += len(batch_results)
            
            # Call batch completion callback if provided
            if on_batch_complete:
                on_batch_complete(batch_results)
            
            # Update progress bar
            if pbar:
                batch_size = len(batches[i]) if i < len(batches) else 0
                pbar.update(batch_size)
        
        # Close progress bar
        if pbar:
            pbar.close()
        
        # Update final metrics
        self.metrics["total_time"] = time.time() - start_time
        
        return results
    
    def _create_batches(self, items: List[T]) -> List[List[T]]:
        """Split items into batches."""
        batches = []
        for i in range(0, len(items), self.config.batch_size):
            batches.append(items[i:i + self.config.batch_size])
        return batches
    
    def _process_batch(self, batch: List[T], process_func: Callable[[T], R]) -> List[R]:
        """Process a batch of items concurrently using ThreadPoolExecutor."""
        results = []
        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            futures = {executor.submit(self._process_with_retry, item, process_func): item for item in batch}
            for future in as_completed(futures, timeout=self.config.timeout_per_batch):
                try:
                    result = future.result()
                    results.append(result)
                    self.metrics["successful_items"] += 1
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    self.metrics["failed_items"] += 1
        
        return results
    
    async def _process_batch_async(self, batch: List[T], process_func: Callable[[T], R]) -> List[R]:
        """Process a batch of items asynchronously."""
        results = []
        tasks = []
        
        for item in batch:
            task = asyncio.create_task(self._process_with_retry_async(item, process_func))
            tasks.append(task)
        
        for task in asyncio.as_completed(tasks, timeout=self.config.timeout_per_batch):
            try:
                result = await task
                results.append(result)
                self.metrics["successful_items"] += 1
            except Exception as e:
                logger.error(f"Error processing item asynchronously: {e}")
                self.metrics["failed_items"] += 1
        
        return results
    
    def _process_with_retry(self, item: T, process_func: Callable[[T], R]) -> R:
        """Process an item with retry logic."""
        for attempt in range(self.config.retry_count):
            try:
                return process_func(item)
            except Exception as e:
                self.metrics["retries"] += 1
                if attempt < self.config.retry_count - 1:
                    logger.warning(f"Retry {attempt + 1}/{self.config.retry_count} after error: {e}")
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Failed after {self.config.retry_count} attempts: {e}")
                    raise
    
    async def _process_with_retry_async(self, item: T, process_func: Callable[[T], R]) -> R:
        """Process an item asynchronously with retry logic."""
        for attempt in range(self.config.retry_count):
            try:
                # If process_func is async, await it, otherwise run in executor
                if asyncio.iscoroutinefunction(process_func):
                    return await process_func(item)
                else:
                    return await asyncio.to_thread(process_func, item)
            except Exception as e:
                self.metrics["retries"] += 1
                if attempt < self.config.retry_count - 1:
                    logger.warning(f"Retry {attempt + 1}/{self.config.retry_count} after error: {e}")
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Failed after {self.config.retry_count} attempts: {e}")
                    raise
    
    def _reset_metrics(self):
        """Reset metrics for a new processing run."""
        self.metrics = {
            "total_items": 0,
            "processed_items": 0,
            "successful_items": 0,
            "failed_items": 0,
            "total_time": 0,
            "batches_processed": 0,
            "retries": 0,
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        metrics = self.metrics.copy()
        
        # Add derived metrics
        if metrics["total_items"] > 0:
            metrics["success_rate"] = metrics["successful_items"] / metrics["total_items"]
        else:
            metrics["success_rate"] = 0
        
        if metrics["total_time"] > 0:
            metrics["items_per_second"] = metrics["processed_items"] / metrics["total_time"]
        else:
            metrics["items_per_second"] = 0
        
        return metrics
