Looking at the PyTorch test suite (#file:test_manifolds.py-2) compared to the JAX implementation (#file:test_manifolds.py), here are the missing tests:

## Missing Tests in JAX Implementation

### 1. **Gyration Tests** (`test_gyration`)
- Tests gyrocommutative law
- Gyrosum inversion law
- Left/right gyroassociative laws
- Möbius addition under gyrations
- Loop properties
- Identity gyroautomorphism property
- First gyrogroup theorems
- **Note**: Only applicable to PoincareBall manifold

### 2. **Enhanced Scalar Multiplication Tests**
The JAX `test_scalar_mul` is missing several checks present in PyTorch:
- **N-Gyroaddition property**: `n ⊗ x = x ⊕ x ⊕ ... ⊕ x` (n times)
- **Distributive laws**: 
  - `(r1 + r2) ⊗ x = (r1 ⊗ x) ⊕ (r2 ⊗ x)`
  - `(-r) ⊗ x = r ⊗ (-x)`
- **Scaling property**: Direction preservation under scaling
- **Homogeneity property**: `‖r ⊗ x‖ = |r| ⊗ ‖x‖`
- **Numerical stability tests**:
  - Multiplication with zero scalars
  - Multiplication with small scalars (near `atol`)
  - Multiplication with large scalars
  - Edge cases with epsilon-norm vectors

### 3. **Enhanced Addition Tests**
The JAX `test_addition` is missing:
- **Gyrotriangle inequality**: `‖x ⊕ y‖ ≤ ‖x‖ ⊕ ‖y‖`
- **Second additive inverse check**: `x ⊕ (-x) ≈ 0`

### 4. **Distance Function Enhancements**
The JAX `test_dist_properties` could add:
- **Version parameter testing** (if applicable in JAX backend)
- Different distance computation methods (e.g., `mobius_direct` for PoincareBall)

### 5. **Exponential/Logarithmic Map Tests**
The JAX tests are more limited than PyTorch:
- **Retraction operation** (`test_expmap_retraction_logmap` in PyTorch)
- **Consistency checks**: `expmap(v, origin) = expmap_0(v)`
- **Inverse operation stability** for arbitrary tangent vectors
- **Numerical stability for finite checks** on expmap/retraction outputs

### 6. **Parallel Transport Enhancements**
The JAX `test_ptransp_preserves_norm` could add:
- **Round-trip stability**: `ptransp(ptransp(v, x, y), y, x) ≈ v`
- **Consistency**: `ptransp(v, origin, x) = ptransp_0(v, x)`
- **Inner product preservation** for multiple vector pairs

### 7. **Tangent Norm Consistency**
The PyTorch suite has `test_tangent_norm` that checks:
- Consistency of `tangent_norm` with `logmap` and `dist`
- Both general and origin-based versions

## Summary Priority

**High Priority:**
1. Gyration tests (PoincareBall-specific)
2. Numerical stability tests for scalar multiplication
3. Enhanced addition properties (gyrotriangle inequality)

**Medium Priority:**
4. Retraction operation tests
5. Tangent norm consistency checks
6. Round-trip parallel transport stability

**Low Priority:**
7. Version parameter testing (if supported)
8. Additional distributive/homogeneity laws

The PyTorch test suite is more comprehensive, especially around numerical edge cases and gyrogroup properties specific to hyperbolic geometry.