# Frontend Development Guidelines

## Component Architecture (`/src/components`)

### Component Creation Rules

1. **When to Create a New Component**
   - Component exceeds 100 lines of code
   - UI element is used in 2+ places
   - Logic complexity requires > 3 hooks
   - Component has multiple internal states
   - Section can be logically isolated (e.g., Header, Footer)
   - Component handles a specific feature (e.g., DocumentUpload)

2. **Component File Structure**
   ```typescript
   // ComponentName.tsx
   import { useState, useEffect } from 'react'
   import type { ComponentProps } from './types'
   
   export const ComponentName: React.FC<ComponentProps> = ({ prop1, prop2 }) => {
     // 1. Hooks
     // 2. Derived State
     // 3. Event Handlers
     // 4. Effects
     // 5. Render
   }
   ```

3. **Component Organization**
   ```
   components/
   ├── feature/           # Feature-specific components
   │   ├── index.ts      # Barrel exports
   │   └── types.ts      # Feature-specific types
   ├── shared/           # Reusable components
   │   ├── Button/
   │   └── Input/
   └── layout/           # Layout components
       ├── Header/
       └── Footer/
   ```

4. **Props Guidelines**
   - Use TypeScript interfaces for props
   - Keep props flat (avoid deep nesting)
   - Use optional props sparingly
   - Document complex props with JSDoc

## App Directory (`/src/app`)

### Page Organization Rules

1. **Route Structure**
   ```
   app/
   ├── layout.tsx        # Root layout
   ├── page.tsx          # Home page
   ├── (auth)/          # Auth group
   │   ├── login/
   │   └── register/
   └── (dashboard)/     # Dashboard group
       ├── layout.tsx   # Dashboard layout
       └── page.tsx     # Dashboard page
   ```

2. **Layout Guidelines**
   - Use nested layouts for shared UI
   - Keep layouts focused on structure
   - Handle metadata in layouts
   - Implement error boundaries

3. **Page Component Rules**
   - Pages should be thin
   - Delegate complex logic to components
   - Use loading.tsx for suspense
   - Implement error.tsx for fallbacks

## Services (`/src/services`)

### API Service Guidelines

1. **api.ts Structure**
   ```typescript
   // api.ts
   import { API_CONFIG } from '@/lib/constants'
   
   class ApiService {
     private static instance: ApiService
     private baseUrl: string
     
     private constructor() {
       this.baseUrl = API_CONFIG.BASE_URL
     }
     
     // Singleton pattern
     public static getInstance(): ApiService {
       if (!ApiService.instance) {
         ApiService.instance = new ApiService()
       }
       return ApiService.instance
     }
     
     // Request wrapper
     private async request<T>(
       endpoint: string,
       options: RequestInit = {}
     ): Promise<T> {
       try {
         const response = await fetch(`${this.baseUrl}${endpoint}`, {
           ...options,
           headers: {
             ...API_CONFIG.DEFAULT_HEADERS,
             ...options.headers,
           },
         })
         
         if (!response.ok) {
           throw await this.handleError(response)
         }
         
         return await response.json()
       } catch (error) {
         this.handleError(error)
       }
     }
     
     // API methods
     public async getData(): Promise<DataType> {
       return this.request<DataType>(API_CONFIG.ENDPOINTS.DATA)
     }
   }
   
   export const api = ApiService.getInstance()
   ```

2. **Error Handling**
   - Centralized error handling
   - Custom error types
   - Retry logic for transient failures
   - Error reporting integration

3. **Request Organization**
   - Group related endpoints
   - Use TypeScript generics
   - Implement request caching
   - Handle request cancellation

## Constants Best Practices (`/src/lib/constants.ts`)

### Structure and Organization

1. **Configuration Groups**
   ```typescript
   export const API_CONFIG = {
     // API-specific configuration
   } as const;

   export const APP_CONFIG = {
     // Application-specific configuration
   } as const;
   ```

   **Rules:**
   - Group related constants into logical namespaces
   - Use PascalCase for configuration object names
   - Export as `const` assertions for type safety
   - Keep groups focused and single-purpose

2. **Environment Variables**
   ```typescript
   BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
   ```

   **Rules:**
   - Always provide fallback values for env variables
   - Use NEXT_PUBLIC_ prefix for client-side env vars
   - Document expected environment variables
   - Keep development-friendly defaults

3. **API Configuration**
   ```typescript
   ENDPOINTS: {
     GENERATE: '/generate',
     DOCUMENTS: '/documents',
     EVALUATE: '/evaluate',
     HEALTH: '/health'
   }
   ```

   **Rules:**
   - Use nested objects for related endpoints
   - Keep endpoint paths normalized (no trailing slashes)
   - Use UPPERCASE for endpoint constants
   - Group endpoints by feature or resource

4. **Default Values**
   ```typescript
   DEFAULT_HEADERS: {
     'Content-Type': 'application/json'
   }
   ```

   **Rules:**
   - Prefix defaults with 'DEFAULT_'
   - Document why defaults were chosen
   - Keep defaults secure and conservative
   - Consider environment impact

5. **Resource Limits**
   ```typescript
   MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
   SUPPORTED_FILE_TYPES: ['pdf', 'txt', 'doc', 'docx'],
   MAX_FILES_PER_REQUEST: 10
   ```

   **Rules:**
   - Include units in comments
   - Use descriptive constant names
   - Keep arrays readonly when possible
   - Document limitations clearly

### Usage Guidelines

1. **When to Create Constants**
   - Configuration values used in multiple places
   - Magic numbers that need explanation
   - Environment-specific values
   - API endpoints and routes
   - Feature flags and toggles
   - Resource limitations
   - Default configurations

2. **Naming Conventions**
   - Use UPPERCASE for static values
   - Use PascalCase for configuration objects
   - Prefix limits with MAX_ or MIN_
   - Prefix defaults with DEFAULT_
   - Use descriptive, specific names

3. **Type Safety**
   ```typescript
   // Good
   export const CONFIG = {
     value: 'string'
   } as const;

   // Better - with explicit type
   export const CONFIG: Readonly<{
     value: string;
   }> = {
     value: 'string'
   } as const;
   ```

   **Rules:**
   - Use `as const` assertions
   - Consider explicit readonly types
   - Avoid type widening
   - Export specific types when needed

4. **Documentation Requirements**
   ```typescript
   /** Maximum file size allowed for upload (10MB) */
   MAX_FILE_SIZE: 10 * 1024 * 1024,

   /** Supported file types for document processing */
   SUPPORTED_FILE_TYPES: ['pdf', 'txt', 'doc', 'docx'] as const,
   ```

   **Rules:**
   - Document units of measurement
   - Explain non-obvious values
   - Include rationale for limits
   - Use JSDoc for complex constants

5. **Organization Patterns**
   ```typescript
   /src/lib/constants/
   ├── api.ts        # API-related constants
   ├── app.ts        # Application constants
   ├── ui.ts         # UI-related constants
   └── index.ts      # Re-exports
   ```

   **Rules:**
   - Split large constant files by domain
   - Use barrel exports for organization
   - Keep related constants together
   - Consider lazy loading for large sets

6. **Maintenance Guidelines**
   - Review and update constants regularly
   - Version control significant changes
   - Keep documentation in sync
   - Consider backwards compatibility
   - Monitor for deprecated values

7. **Security Considerations**
   - Never store secrets in constants
   - Use environment variables for sensitive data
   - Validate environment variables
   - Consider exposure in client bundles

## Library (`/src/lib`)

### Utility Organization Rules

1. **Constants Structure**
   ```typescript
   // constants/index.ts
   export * from './api'
   export * from './ui'
   export * from './validation'
   
   // constants/api.ts
   export const API_CONSTANTS = {/*...*/}
   
   // constants/ui.ts
   export const UI_CONSTANTS = {/*...*/}
   ```

2. **Helpers Organization**
   ```typescript
   // helpers/index.ts
   export * from './date'
   export * from './string'
   export * from './validation'
   ```

3. **Hooks Organization**
   ```typescript
   // hooks/index.ts
   export * from './useApi'
   export * from './useAuth'
   export * from './useForm'
   ```

### When to Create New Utilities

1. **Create Constants When**
   - Value is used in 2+ places
   - Configuration might change
   - Value needs documentation
   - Environment-specific values

2. **Create Helpers When**
   - Logic is reused
   - Complex transformations
   - Business logic abstraction
   - Type conversions

3. **Create Hooks When**
   - State logic is reused
   - Complex state management
   - Side effect patterns
   - Browser API abstractions

## Types (`/src/types`)

### Type Organization Rules

1. **File Structure**
   ```typescript
   // types/api.ts
   export interface ApiResponse<T> {
     data: T
     meta: ResponseMetadata
   }
   
   // types/models.ts
   export interface User {/*...*/}
   
   // types/components.ts
   export interface ButtonProps {/*...*/}
   ```

2. **Type Creation Guidelines**
   - Create types for API responses
   - Share types between components
   - Use generics for flexibility
   - Document complex types

3. **Type Usage Rules**
   - Prefer interfaces for objects
   - Use type for unions/intersections
   - Export shared types
   - Keep types DRY

### Best Practices for Types

1. **Naming Conventions**
   - PascalCase for interfaces/types
   - Suffix props with 'Props'
   - Prefix state with 'State'
   - Use descriptive names

2. **Type Organization**
   ```
   types/
   ├── api/              # API related types
   ├── components/       # Component props
   ├── models/          # Business models
   └── utils/           # Utility types
   ```

3. **Type Safety Rules**
   - Avoid 'any'
   - Use strict null checks
   - Define enum values
   - Use readonly when appropriate

## State Management

### When to Use Different State Solutions

1. **Local State (useState)**
   - Component-specific data
   - Simple toggle states
   - Form input values
   - UI state

2. **Context**
   - Theme data
   - User preferences
   - Authentication state
   - Shared configuration

3. **Global State**
   - Complex app state
   - Cached API data
   - Cross-component state
   - Performance-critical data

## Performance Guidelines

1. **Component Optimization**
   - Use React.memo for expensive renders
   - Implement useMemo for complex calculations
   - Use useCallback for function props
   - Lazy load components

2. **Data Fetching**
   - Implement request caching
   - Use SWR/React Query
   - Handle loading states
   - Optimize payload size

3. **Bundle Optimization**
   - Code splitting
   - Tree shaking
   - Dynamic imports
   - Image optimization

## Error Handling

1. **Component Error Boundaries**
   ```typescript
   // components/ErrorBoundary.tsx
   class ErrorBoundary extends React.Component<Props, State> {
     static getDerivedStateFromError(error: Error) {
       return { hasError: true, error }
     }
     
     componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
       // Log error to service
     }
     
     render() {
       if (this.state.hasError) {
         return <ErrorFallback error={this.state.error} />
       }
       return this.props.children
     }
   }
   ```

2. **API Error Handling**
   - Custom error types
   - Retry mechanisms
   - User-friendly messages
   - Error reporting

## Testing Guidelines

1. **Component Testing**
   - Unit tests for logic
   - Integration tests for features
   - Snapshot tests for UI
   - Accessibility tests

2. **Test Organization**
   ```
   __tests__/
   ├── components/
   ├── hooks/
   └── utils/
   ```

3. **Testing Best Practices**
   - Test business logic
   - Mock external dependencies
   - Use testing library
   - Write meaningful assertions 

## TypeScript Types Organization (`/src/types/index.ts`)

### Section Organization

1. **Type Grouping Pattern**
   ```typescript
   // =============================================================================
   // CORE BACKEND TYPES
   // =============================================================================
   
   // Backend model types here
   
   // =============================================================================
   // API REQUEST/RESPONSE TYPES
   // =============================================================================
   
   // API-specific types here
   ```

   **Rules:**
   - Use clear section dividers with descriptive headers
   - Group related types together
   - Order sections by dependency flow
   - Comment section purposes

2. **Backend Model Types**
   ```typescript
   export interface DocumentInput {
     content: string;
     metadata: Record<string, unknown>;
     source?: string;
   }

   export type EvolutionType = 
     | "simple_evolution" 
     | "multi_context_evolution" 
     | "reasoning_evolution" 
     | "complex_evolution";
   ```

   **Rules:**
   - Match FastAPI/backend models exactly
   - Use strict typing (avoid `any`)
   - Document backend dependencies
   - Keep types synchronized with API

3. **API Contract Types**
   ```typescript
   export interface APIResponse<T> {
     success: boolean;
     data?: T;
     error?: APIError;
   }

   export interface APIError {
     detail: string;
     status_code?: number;
   }
   ```

   **Rules:**
   - Use generics for flexible responses
   - Consistent error handling types
   - Optional fields for nullable data
   - Include status codes and metadata

4. **Frontend-Specific Types**
   ```typescript
   export interface DisplayQuestion {
     id: string;
     question: string;
     answer: string;
     context: (string | EnhancedContext)[];
     level: string;
     metadata?: {
       confidence?: number;
       source?: string;
       complexity_level?: number;
       evolution_type?: EvolutionType;
     };
   }
   ```

   **Rules:**
   - Transform backend types for UI needs
   - Include UI-specific metadata
   - Support multiple data formats
   - Document type transformations

5. **Component Props Types**
   ```typescript
   export interface DocumentUploadProps {
     documents: UploadedDocument[];
     setDocuments: (docs: UploadedDocument[]) => void;
     onNext: () => void;
   }
   ```

   **Rules:**
   - Include all required props
   - Type event handlers properly
   - Document optional props
   - Use callback signatures

6. **Utility Types**
   ```typescript
   export type GenerationStatus = 
     | "idle" 
     | "uploading" 
     | "processing" 
     | "generating" 
     | "evaluating" 
     | "completed" 
     | "error";
   ```

   **Rules:**
   - Use union types for states
   - Keep enums consistent
   - Document type constraints
   - Use descriptive names

7. **Helper Function Types**
   ```typescript
   export interface ConversionHelpers {
     frontendToBackendDocument: (doc: UploadedDocument) => DocumentInput;
     frontendToBackendSettings: (settings: FrontendGenerationSettings) => GenerationSettings;
   }
   ```

   **Rules:**
   - Type function signatures clearly
   - Group related conversions
   - Document transformations
   - Include return types

### Type Safety Guidelines

1. **Type Assertions**
   ```typescript
   // Prefer
   interface EnhancedContext {
     text: string;
     source: string;
     document_index: number;
   }

   // Over
   type EnhancedContext = {
     text: string;
     source: string;
     document_index: number;
   }
   ```

   **Rules:**
   - Use interfaces for objects
   - Use type for unions/intersections
   - Avoid type assertions when possible
   - Document type decisions

2. **Generic Types**
   ```typescript
   export interface APIResponse<T> {
     success: boolean;
     data?: T;
     error?: APIError;
   }
   ```

   **Rules:**
   - Use generics for reusable types
   - Constrain generics when needed
   - Document generic parameters
   - Keep generic names meaningful

3. **Record Types**
   ```typescript
   metadata: Record<string, unknown>;
   ```

   **Rules:**
   - Use Record for dynamic objects
   - Avoid index signatures when possible
   - Type values appropriately
   - Document expected keys

4. **Union Types**
   ```typescript
   export type GenerationStep = "upload" | "generate" | "results";
   ```

   **Rules:**
   - Use literal unions for enums
   - Keep values consistent
   - Document valid values
   - Consider exhaustiveness checks

### Maintenance Best Practices

1. **Type Updates**
   - Keep types synchronized with backend
   - Document breaking changes
   - Version control type changes
   - Test type compatibility

2. **Documentation**
   - Document complex types
   - Include usage examples
   - Note type dependencies
   - Explain type decisions

3. **Organization**
   - Group related types
   - Use clear naming
   - Keep files focused
   - Consider splitting large files

4. **Type Exports**
   - Export all public types
   - Use barrel exports
   - Document public API
   - Consider type visibility 

## Next.js Configuration Best Practices

### Configuration Files Overview

1. **next.config.ts**
   ```typescript
   import type { NextConfig } from "next";

   const nextConfig: NextConfig = {
     reactStrictMode: true,
     images: {
       domains: ['your-image-domain.com'],
     },
     async headers() {
       return [
         {
           source: '/(.*)',
           headers: [
             {
               key: 'X-DNS-Prefetch-Control',
               value: 'on'
             },
           ],
         },
       ];
     },
     // Environment variables that need to be exposed to the browser
     env: {
       customKey: process.env.CUSTOM_KEY,
     },
   };

   export default nextConfig;
   ```

   **Rules:**
   - Use TypeScript for configuration
   - Enable strict mode for better error detection
   - Configure image domains for next/image
   - Set up security headers
   - Document environment variables

2. **vercel.json**
   ```json
   {
     "name": "evolsynth-frontend",
     "version": 2,
     "framework": "nextjs",
     "buildCommand": "npm run build",
     "outputDirectory": ".next",
     "installCommand": "npm install",
     "devCommand": "npm run dev",
     "env": {
       "NEXT_PUBLIC_API_URL": "https://evolsynth-api-production.up.railway.app"
     },
     "build": {
       "env": {
         "NEXT_PUBLIC_API_URL": "https://evolsynth-api-production.up.railway.app"
       }
     },
     "functions": {
       "app/api/**/*.ts": {
         "maxDuration": 30
       }
     },
     "headers": [
       {
         "source": "/(.*)",
         "headers": [
           {
             "key": "X-Frame-Options",
             "value": "DENY"
           },
           {
             "key": "X-Content-Type-Options",
             "value": "nosniff"
           },
           {
             "key": "Referrer-Policy",
             "value": "strict-origin-when-cross-origin"
           }
         ]
       }
     ]
   }
   ```

   **Rules:**
   - Set descriptive project name
   - Configure build settings
   - Define environment variables
   - Set up security headers
   - Configure serverless function limits
   - Document deployment settings

3. **Environment Configuration**
   ```plaintext
   # .env.local
   NEXT_PUBLIC_API_URL=http://localhost:8000

   # .env.development
   NEXT_PUBLIC_API_URL=http://localhost:8000

   # .env.production
   NEXT_PUBLIC_API_URL=https://api.production.com
   ```

   **Rules:**
   - Use environment-specific files
   - Prefix client-side variables with NEXT_PUBLIC_
   - Include .env.example for documentation
   - Keep sensitive data out of version control
   - Document required variables

### Security Best Practices

1. **Headers Configuration**
   ```typescript
   // next.config.ts
   const securityHeaders = [
     {
       key: 'X-DNS-Prefetch-Control',
       value: 'on'
     },
     {
       key: 'X-XSS-Protection',
       value: '1; mode=block'
     },
     {
       key: 'X-Frame-Options',
       value: 'DENY'
     },
     {
       key: 'X-Content-Type-Options',
       value: 'nosniff'
     },
     {
       key: 'Referrer-Policy',
       value: 'strict-origin-when-cross-origin'
     }
   ];
   ```

   **Rules:**
   - Implement security headers
   - Prevent clickjacking
   - Enable XSS protection
   - Control resource loading
   - Configure CSP when needed

2. **API Route Protection**
   ```typescript
   // next.config.ts
   {
     async rewrites() {
       return [
         {
           source: '/api/:path*',
           destination: 'https://api.backend.com/:path*',
         },
       ];
     },
   }
   ```

   **Rules:**
   - Configure API routes securely
   - Use rewrites for proxy
   - Rate limit API routes
   - Validate request origins
   - Handle CORS properly

### Performance Optimization

1. **Build Configuration**
   ```typescript
   // next.config.ts
   {
     swcMinify: true,
     compiler: {
       removeConsole: process.env.NODE_ENV === 'production',
     },
     experimental: {
       optimizeCss: true,
     },
   }
   ```

   **Rules:**
   - Enable SWC minification
   - Remove console in production
   - Optimize CSS
   - Configure chunking
   - Enable compression

2. **Image Optimization**
   ```typescript
   // next.config.ts
   {
     images: {
       domains: ['trusted-domain.com'],
       deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
       imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
       formats: ['image/webp'],
     },
   }
   ```

   **Rules:**
   - Configure image domains
   - Set appropriate sizes
   - Enable WebP format
   - Optimize loading
   - Cache properly

### Development Experience

1. **TypeScript Configuration**
   ```typescript
   // next.config.ts
   {
     typescript: {
       ignoreBuildErrors: false,
       tsconfigPath: './tsconfig.json',
     },
   }
   ```

   **Rules:**
   - Enforce type checking
   - Configure paths
   - Set strict mode
   - Handle type errors
   - Document type decisions

2. **Development Tools**
   ```json
   // package.json
   {
     "scripts": {
       "dev": "next dev",
       "build": "next build",
       "start": "next start",
       "lint": "next lint",
       "type-check": "tsc --noEmit"
     }
   }
   ```

   **Rules:**
   - Include type checking
   - Configure linting
   - Set up formatting
   - Add build scripts
   - Document commands

### Deployment Configuration

1. **Vercel-Specific Settings**
   ```json
   // vercel.json
   {
     "functions": {
       "app/api/**/*.ts": {
         "maxDuration": 30
       }
     },
     "regions": ["cdg1"],
     "public": false
   }
   ```

   **Rules:**
   - Configure function limits
   - Set deployment regions
   - Handle environment variables
   - Configure build settings
   - Set up redirects

2. **CI/CD Integration**
   ```yaml
   # .github/workflows/deploy.yml
   name: Deploy
   on: [push]
   jobs:
     deploy:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - uses: actions/setup-node@v2
         - run: npm ci
         - run: npm run build
         - uses: vercel/actions/cli@v2
   ```

   **Rules:**
   - Automate deployments
   - Run tests before deploy
   - Configure environment
   - Handle secrets
   - Document process 