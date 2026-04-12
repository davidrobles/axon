package net.davidrobles.axon.agents;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import net.davidrobles.axon.Experience;

/**
 * A fixed-capacity circular replay buffer for storing and sampling experiences.
 *
 * <p>New experiences are added in O(1). When the buffer is full, the oldest experience is
 * overwritten. Random sampling without replacement is supported for mini-batch updates.
 *
 * @param <S> the state type
 * @param <A> the action type
 */
public class ReplayBuffer<S, A> {
    private final int capacity;
    private final List<Experience<S, A>> buffer;
    private int writeIndex = 0;
    private int size = 0;

    /**
     * @param capacity maximum number of experiences to store; must be >= 1
     */
    public ReplayBuffer(int capacity) {
        if (capacity < 1)
            throw new IllegalArgumentException("capacity must be >= 1, got: " + capacity);
        this.capacity = capacity;
        this.buffer = new ArrayList<>(capacity);
    }

    /**
     * Adds an experience to the buffer. If the buffer is full, the oldest experience is
     * overwritten.
     *
     * @param experience the experience to store
     */
    public void add(Experience<S, A> experience) {
        Objects.requireNonNull(experience, "experience must not be null");
        if (size < capacity) {
            buffer.add(experience);
            size++;
        } else {
            buffer.set(writeIndex, experience);
        }
        writeIndex = (writeIndex + 1) % capacity;
    }

    /**
     * Returns a random sample of {@code batchSize} experiences without replacement.
     *
     * @param batchSize number of experiences to sample; must be <= {@link #size()}
     * @param rng random number generator
     * @return an unordered list of sampled experiences
     * @throws IllegalArgumentException if {@code batchSize} exceeds the current buffer size
     */
    public List<Experience<S, A>> sample(int batchSize, Random rng) {
        if (batchSize > size)
            throw new IllegalArgumentException(
                    "batchSize (" + batchSize + ") exceeds buffer size (" + size + ")");
        // Fisher-Yates partial shuffle to select batchSize indices without replacement
        List<Integer> indices = new ArrayList<>(size);
        for (int i = 0; i < size; i++) indices.add(i);
        List<Experience<S, A>> batch = new ArrayList<>(batchSize);
        for (int i = 0; i < batchSize; i++) {
            int j = i + rng.nextInt(size - i);
            int tmp = indices.get(i);
            indices.set(i, indices.get(j));
            indices.set(j, tmp);
            batch.add(buffer.get(indices.get(i)));
        }
        return batch;
    }

    /** Returns the number of experiences currently stored. */
    public int size() {
        return size;
    }

    /** Returns {@code true} if the buffer has reached its capacity. */
    public boolean isFull() {
        return size == capacity;
    }
}
