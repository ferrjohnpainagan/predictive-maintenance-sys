# NestJS API Gateway Free Tier Implementation Plan - Predictive Maintenance System

## Project Overview

Implementation plan for the NestJS API Gateway using AWS Free Tier resources. This approach focuses on EC2-based deployment instead of AWS Fargate to stay within free tier limits while maintaining full functionality.

## Technology Stack (Free Tier)

- **Framework**: NestJS (Latest version)
- **Language**: TypeScript
- **Database**: Supabase (Free tier)
- **Authentication**: Supabase Auth (JWT)
- **Communication**: gRPC (for ML service), REST (for frontend)
- **Documentation**: OpenAPI/Swagger
- **Testing**: Jest, Supertest
- **Logging**: Pino with CloudWatch (Free tier)
- **Deployment**: EC2 t2.micro (Free tier eligible)

## Free Tier Considerations

- **EC2**: Using t2.micro instances (1 vCPU, 1GB RAM)
- **Supabase**: Free tier (2 databases, 500MB storage)
- **CloudWatch**: Basic monitoring and logging (Free tier)
- **Cost**: $0/month for 12 months

---

## Phase 1: Project Setup and Foundation

**Duration**: 2-3 days  
**Priority**: Critical

### Objectives

- Initialize NestJS project with TypeScript
- Configure development environment optimized for free tier
- Set up project structure and conventions

### Tasks

1. **Project Initialization**

   - [ ] Create NestJS project with CLI
   - [ ] Configure TypeScript (strict mode)
   - [ ] Set up ESLint and Prettier
   - [ ] Configure Git repository
   - [ ] Set up environment configuration for free tier

2. **Dependencies Installation (Free Tier Optimized)**

   ```bash
   # Core dependencies
   npm install @nestjs/config @nestjs/swagger
   npm install @supabase/supabase-js
   npm install @grpc/grpc-js @grpc/proto-loader
   npm install pino pino-pretty nestjs-pino
   npm install class-validator class-transformer
   npm install @nestjs/throttler helmet

   # Free tier specific
   npm install compression
   npm install @nestjs/serve-static
   ```

3. **Project Structure (Free Tier Focused)**

   ```
   backend/api-gateway/
   ├── src/
   │   ├── auth/               # Supabase auth integration
   │   ├── config/             # Environment configs
   │   ├── engine/             # Engine management
   │   ├── predictions/        # ML predictions API
   │   ├── health/             # Health checks for EC2
   │   ├── grpc/               # gRPC client configuration
   │   ├── common/             # Shared utilities
   │   ├── interceptors/       # Logging, compression
   │   ├── guards/             # Auth guards
   │   ├── dto/                # Data transfer objects
   │   └── main.ts             # Application entry point
   ├── deployment/
   │   ├── docker/
   │   │   └── Dockerfile.free-tier    # Optimized for t2.micro
   │   ├── scripts/
   │   │   ├── deploy-ec2.sh           # EC2 deployment script
   │   │   └── health-check.sh         # Health monitoring
   │   └── systemd/
   │       └── api-gateway.service     # Service configuration
   ├── test/                   # E2E tests
   ├── docs/                   # API documentation
   └── package.json
   ```

4. **Environment Configuration**
   ```typescript
   // src/config/configuration.ts
   export default () => ({
     port: parseInt(process.env.PORT, 10) || 3000,
     nodeEnv: process.env.NODE_ENV || "development",

     // Supabase configuration (free tier)
     supabase: {
       url: process.env.SUPABASE_URL,
       key: process.env.SUPABASE_ANON_KEY,
       serviceKey: process.env.SUPABASE_SERVICE_KEY,
     },

     // gRPC configuration for EC2
     grpc: {
       mlServiceUrl: process.env.ML_SERVICE_URL || "localhost:50051",
       timeout: 5000,
     },

     // Free tier optimizations
     performance: {
       enableCompression: true,
       maxRequestSize: "1mb",
       rateLimitTtl: 60,
       rateLimitLimit: 100,
     },
   })
   ```

### Deliverables

- [ ] NestJS project initialized
- [ ] Development environment configured for free tier
- [ ] Project structure established
- [ ] Basic configuration completed

---

## Phase 2: Core Infrastructure

**Duration**: 3-4 days  
**Priority**: High

### Objectives

- Set up Supabase integration
- Configure authentication and authorization
- Implement basic health monitoring

### Tasks

1. **Supabase Integration**

   ```typescript
   // src/config/supabase.config.ts
   import { createClient } from "@supabase/supabase-js"

   export const createSupabaseClient = () => {
     const supabaseUrl = process.env.SUPABASE_URL!
     const supabaseKey = process.env.SUPABASE_ANON_KEY!

     return createClient(supabaseUrl, supabaseKey, {
       auth: {
         autoRefreshToken: true,
         persistSession: true,
       },
       // Free tier optimizations
       db: {
         schema: "public",
       },
       global: {
         headers: { "x-my-custom-header": "my-app-name" },
       },
     })
   }
   ```

2. **Authentication Module (Free Tier)**

   ```typescript
   // src/auth/auth.service.ts
   @Injectable()
   export class AuthService {
     private supabase = createSupabaseClient()

     async validateUser(token: string): Promise<any> {
       try {
         const {
           data: { user },
           error,
         } = await this.supabase.auth.getUser(token)

         if (error || !user) {
           throw new UnauthorizedException("Invalid token")
         }

         return user
       } catch (error) {
         // Handle free tier rate limits gracefully
         if (error.status === 429) {
           throw new HttpException(
             "Rate limit exceeded",
             HttpStatus.TOO_MANY_REQUESTS
           )
         }
         throw error
       }
     }
   }
   ```

3. **Health Check System (EC2 Optimized)**
   ```typescript
   // src/health/health.controller.ts
   @Controller("health")
   export class HealthController {
     constructor(private readonly healthCheckService: HealthCheckService) {}

     @Get()
     @HealthCheck()
     check() {
       return this.healthCheckService.check([
         // Database connectivity
         () => this.healthCheckService.pingCheck("supabase", this.supabaseUrl),

         // ML Service connectivity
         () =>
           this.healthCheckService.pingCheck("ml-service", this.mlServiceUrl),

         // Memory usage (important for t2.micro)
         () =>
           this.healthCheckService.memoryHealthIndicator.checkHeap(
             "memory_heap",
             200 * 1024 * 1024
           ),

         // Disk usage
         () =>
           this.healthCheckService.diskHealthIndicator.checkStorage("storage", {
             thresholdPercent: 0.8,
             path: "/",
           }),
       ])
     }
   }
   ```

### Deliverables

- [ ] Supabase integration configured
- [ ] Authentication system implemented
- [ ] Health monitoring established

---

## Phase 3: gRPC Client Configuration (Free Tier)

**Duration**: 3-4 days  
**Priority**: High

### Objectives

- Set up gRPC client for ML service communication
- Optimize for EC2-to-EC2 communication
- Implement error handling and retry logic

### Tasks

1. **gRPC Client Setup**

   ```typescript
   // src/grpc/ml-service.client.ts
   import * as grpc from "@grpc/grpc-js"
   import * as protoLoader from "@grpc/proto-loader"

   @Injectable()
   export class MLServiceClient {
     private client: any

     constructor() {
       const packageDefinition = protoLoader.loadSync("ml_service.proto", {
         keepCase: true,
         longs: String,
         enums: String,
         defaults: true,
         oneofs: true,
       })

       const protoDescriptor = grpc.loadPackageDefinition(packageDefinition)

       // Use internal EC2 IP for free tier (no load balancer costs)
       this.client = new protoDescriptor.MLService(
         process.env.ML_SERVICE_INTERNAL_URL || "localhost:50051",
         grpc.credentials.createInsecure()
       )
     }

     async predictRUL(sensorData: any): Promise<any> {
       return new Promise((resolve, reject) => {
         // Set timeout appropriate for t2.micro performance
         const deadline = new Date()
         deadline.setSeconds(deadline.getSeconds() + 30)

         this.client.predictRUL(sensorData, { deadline }, (error, response) => {
           if (error) {
             // Handle EC2 connectivity issues
             if (error.code === grpc.status.UNAVAILABLE) {
               reject(new ServiceUnavailableException("ML Service unavailable"))
             }
             reject(error)
           }
           resolve(response)
         })
       })
     }
   }
   ```

2. **Protocol Buffer Definitions**

   ```protobuf
   // proto/ml_service.proto
   syntax = "proto3";

   package ml_service;

   service MLService {
     rpc PredictRUL (PredictionRequest) returns (PredictionResponse);
     rpc GetModelInfo (Empty) returns (ModelInfo);
     rpc HealthCheck (Empty) returns (HealthStatus);
   }

   message PredictionRequest {
     string engine_id = 1;
     repeated SensorReading sensors = 2;
   }

   message SensorReading {
     string sensor_id = 1;
     double value = 2;
     int64 timestamp = 3;
   }

   message PredictionResponse {
     string engine_id = 1;
     double predicted_rul = 2;
     double confidence = 3;
     string model_version = 4;
   }
   ```

### Deliverables

- [ ] gRPC client configured
- [ ] Protocol buffers defined
- [ ] Error handling implemented

---

## Phase 4: API Endpoints (Free Tier Optimized)

**Duration**: 5-6 days  
**Priority**: High

### Objectives

- Implement REST API endpoints
- Add request/response validation
- Optimize for free tier performance

### Tasks

1. **Engine Management API**

   ```typescript
   // src/engine/engine.controller.ts
   @Controller("engines")
   @UseGuards(AuthGuard)
   export class EngineController {
     constructor(
       private readonly engineService: EngineService,
       private readonly cacheManager: CacheManager // For performance
     ) {}

     @Get()
     @UseInterceptors(CacheInterceptor)
     @CacheTTL(300) // 5 minutes cache for free tier optimization
     async getEngines(@Query() query: GetEnginesDto) {
       return this.engineService.getEngines(query)
     }

     @Get(":id")
     @UseInterceptors(CacheInterceptor)
     async getEngine(@Param("id") id: string) {
       return this.engineService.getEngine(id)
     }
   }
   ```

2. **Predictions API (Free Tier)**
   ```typescript
   // src/predictions/predictions.controller.ts
   @Controller("predictions")
   @UseGuards(AuthGuard)
   @UseInterceptors(CompressionInterceptor) // Reduce bandwidth
   export class PredictionsController {
     constructor(
       private readonly predictionsService: PredictionsService,
       private readonly mlServiceClient: MLServiceClient
     ) {}

     @Post()
     @UseInterceptors(TimeoutInterceptor)
     @HttpCode(HttpStatus.OK)
     async predictRUL(@Body() predictionDto: PredictionRequestDto) {
       try {
         // Rate limiting for free tier
         await this.predictionsService.checkRateLimit(predictionDto.engine_id)

         const result = await this.mlServiceClient.predictRUL(predictionDto)

         // Store result in Supabase (free tier)
         await this.predictionsService.storePrediction(result)

         return result
       } catch (error) {
         // Handle free tier limitations gracefully
         this.logger.error("Prediction failed", error)
         throw new BadRequestException(
           "Prediction service temporarily unavailable"
         )
       }
     }
   }
   ```

### Deliverables

- [ ] REST API endpoints implemented
- [ ] Request validation configured
- [ ] Performance optimizations applied

---

## Phase 5: Testing (Free Tier)

**Duration**: 4-5 days  
**Priority**: Medium

### Objectives

- Set up unit and integration tests
- Configure testing for free tier constraints
- Implement performance tests for t2.micro

### Tasks

1. **Unit Tests**

   ```typescript
   // test/auth/auth.service.spec.ts
   describe("AuthService", () => {
     let service: AuthService

     beforeEach(async () => {
       const module: TestingModule = await Test.createTestingModule({
         providers: [
           AuthService,
           {
             provide: "SUPABASE_CLIENT",
             useValue: mockSupabaseClient,
           },
         ],
       }).compile()

       service = module.get<AuthService>(AuthService)
     })

     it("should validate user token", async () => {
       // Test implementation
     })
   })
   ```

2. **Integration Tests (Free Tier Focused)**
   ```typescript
   // test/app.e2e-spec.ts
   describe("AppController (e2e) - Free Tier", () => {
     let app: INestApplication

     beforeEach(async () => {
       const moduleFixture: TestingModule = await Test.createTestingModule({
         imports: [AppModule],
       }).compile()

       app = moduleFixture.createNestApplication()

       // Configure for t2.micro constraints
       app.use(compression())
       app.setGlobalPipes(
         new ValidationPipe({
           transform: true,
           whitelist: true,
         })
       )

       await app.init()
     })

     it("/health (GET) - should return healthy status", () => {
       return request(app.getHttpServer())
         .get("/health")
         .expect(200)
         .expect((res) => {
           expect(res.body.status).toBe("ok")
         })
     })
   })
   ```

### Deliverables

- [ ] Unit tests implemented
- [ ] Integration tests configured
- [ ] Performance testing completed

---

## Phase 6: Free Tier Deployment Configuration

**Duration**: 4-5 days  
**Priority**: Critical

### Objectives

- Configure Docker for EC2 deployment
- Set up deployment scripts
- Implement monitoring for t2.micro

### Tasks

1. **Docker Configuration (Free Tier Optimized)**

   ```dockerfile
   # Dockerfile.free-tier
   FROM node:18-alpine AS base

   # Install dependencies only when needed
   FROM base AS deps
   RUN apk add --no-cache libc6-compat
   WORKDIR /app

   COPY package*.json ./
   # Use npm ci for faster, reliable builds
   RUN npm ci --only=production && npm cache clean --force

   # Rebuild the source code only when needed
   FROM base AS builder
   WORKDIR /app
   COPY --from=deps /app/node_modules ./node_modules
   COPY . .

   # Build with production optimizations
   RUN npm run build

   # Production image optimized for t2.micro
   FROM base AS runner
   WORKDIR /app

   ENV NODE_ENV production

   RUN addgroup --system --gid 1001 nodejs
   RUN adduser --system --uid 1001 nestjs

   COPY --from=builder --chown=nestjs:nodejs /app/dist ./dist
   COPY --from=builder --chown=nestjs:nodejs /app/node_modules ./node_modules
   COPY --from=builder --chown=nestjs:nodejs /app/package.json ./package.json

   USER nestjs

   EXPOSE 3000

   ENV PORT 3000
   ENV HOSTNAME "0.0.0.0"

   # Health check for EC2
   HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
     CMD curl -f http://localhost:3000/health || exit 1

   CMD ["node", "dist/main"]
   ```

2. **EC2 Deployment Scripts**

   ```bash
   #!/bin/bash
   # deployment/scripts/deploy-ec2.sh

   set -e

   echo "🚀 Deploying API Gateway to EC2 (Free Tier)"

   # Variables
   APP_NAME="predictive-maintenance-api"
   SERVICE_PORT=3000
   DOCKER_IMAGE="$APP_NAME:latest"

   # Build optimized image for t2.micro
   echo "📦 Building Docker image..."
   docker build -f deployment/docker/Dockerfile.free-tier -t $DOCKER_IMAGE .

   # Stop existing container
   echo "🛑 Stopping existing container..."
   docker stop $APP_NAME || true
   docker rm $APP_NAME || true

   # Run new container with resource limits for t2.micro
   echo "🏃 Starting new container..."
   docker run -d \
     --name $APP_NAME \
     --restart unless-stopped \
     -p $SERVICE_PORT:3000 \
     --memory=800m \
     --cpus=0.8 \
     --env-file .env.production \
     $DOCKER_IMAGE

   # Wait for health check
   echo "🔍 Waiting for health check..."
   for i in {1..30}; do
     if curl -f http://localhost:$SERVICE_PORT/health > /dev/null 2>&1; then
       echo "✅ Service is healthy!"
       break
     fi
     echo "⏳ Waiting for service to be ready... ($i/30)"
     sleep 2
   done

   echo "🎉 Deployment completed!"
   ```

3. **Systemd Service Configuration**

   ```ini
   # deployment/systemd/api-gateway.service
   [Unit]
   Description=Predictive Maintenance API Gateway
   After=docker.service
   Requires=docker.service

   [Service]
   Type=oneshot
   RemainAfterExit=yes
   ExecStart=/opt/api-gateway/scripts/deploy-ec2.sh
   ExecStop=/usr/bin/docker stop predictive-maintenance-api
   TimeoutStartSec=0
   Restart=on-failure
   RestartSec=30s

   [Install]
   WantedBy=multi-user.target
   ```

### Deliverables

- [ ] Docker configuration optimized for t2.micro
- [ ] Deployment scripts created
- [ ] Service configuration implemented

---

## Phase 7: Performance Optimization (Free Tier)

**Duration**: 3-4 days  
**Priority**: Medium

### Objectives

- Optimize for t2.micro performance constraints
- Implement caching strategies
- Monitor resource usage

### Tasks

1. **Memory Optimization**

   ```typescript
   // src/common/interceptors/memory.interceptor.ts
   @Injectable()
   export class MemoryOptimizationInterceptor implements NestInterceptor {
     private readonly logger = new Logger(MemoryOptimizationInterceptor.name)

     intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
       const memBefore = process.memoryUsage()

       return next.handle().pipe(
         tap(() => {
           const memAfter = process.memoryUsage()
           const memDiff = memAfter.heapUsed - memBefore.heapUsed

           // Log memory usage for t2.micro monitoring
           if (memDiff > 50 * 1024 * 1024) {
             // 50MB threshold
             this.logger.warn(
               `High memory usage detected: ${memDiff / 1024 / 1024}MB`
             )
           }

           // Force garbage collection if memory usage is high
           if (memAfter.heapUsed > 700 * 1024 * 1024) {
             // 700MB threshold for t2.micro
             if (global.gc) {
               global.gc()
             }
           }
         })
       )
     }
   }
   ```

2. **Caching Strategy**

   ```typescript
   // src/config/cache.config.ts
   import { CacheModule } from "@nestjs/cache-manager"
   import * as redisStore from "cache-manager-redis-store"

   export const CacheConfig = CacheModule.register({
     store: "memory", // Use memory cache for free tier (no Redis costs)
     max: 1000, // Maximum number of items in cache
     ttl: 300, // 5 minutes default TTL

     // Free tier optimization
     isGlobal: true,
   })
   ```

### Deliverables

- [ ] Performance optimizations implemented
- [ ] Caching configured
- [ ] Resource monitoring established

---

## Migration Strategy to Paid Services

### Phase 1: Preparation (Month 10-11)

1. **Containerization Review**

   - [ ] Ensure Docker images are production-ready
   - [ ] Update resource limits for ECS Fargate
   - [ ] Test with higher memory/CPU allocations

2. **Configuration Updates**
   ```typescript
   // src/config/production.config.ts
   export const ProductionConfig = {
     // ECS Fargate configuration
     server: {
       port: parseInt(process.env.PORT, 10) || 3000,
       cors: {
         origin: process.env.FRONTEND_URL,
         credentials: true,
       },
     },

     // Redis cache for production
     cache: {
       store: "redis",
       host: process.env.REDIS_HOST,
       port: parseInt(process.env.REDIS_PORT, 10) || 6379,
       ttl: 600,
     },

     // Production logging
     logging: {
       level: "info",
       format: "json",
       destination: "cloudwatch",
     },
   }
   ```

### Phase 2: Migration Execution (Month 12)

1. **Infrastructure Migration**

   - [ ] Deploy to ECS Fargate using existing infrastructure plan
   - [ ] Configure Application Load Balancer
   - [ ] Set up auto-scaling policies

2. **Data Migration**
   - [ ] Backup Supabase data
   - [ ] Migrate to production Supabase instance
   - [ ] Update connection strings

### Phase 3: Validation (Month 12+)

1. **Performance Testing**

   - [ ] Load testing with production resources
   - [ ] Monitor auto-scaling behavior
   - [ ] Validate high availability

2. **Monitoring Enhancement**
   - [ ] Enable detailed CloudWatch metrics
   - [ ] Configure X-Ray tracing
   - [ ] Set up advanced alerting

---

## Success Metrics (Free Tier)

### Performance Targets

- **Response Time**: < 500ms (P95)
- **Memory Usage**: < 800MB (t2.micro limit)
- **CPU Usage**: < 80% (sustained)
- **Availability**: > 95%

### Operational Targets

- **Deployment Time**: < 5 minutes
- **Recovery Time**: < 10 minutes
- **Cost**: $0/month (within free tier)

---

## Timeline Summary

| Phase                        | Duration | Dependencies          |
| ---------------------------- | -------- | --------------------- |
| Phase 1: Project Setup       | 2-3 days | AWS Free Tier Account |
| Phase 2: Core Infrastructure | 3-4 days | Phase 1               |
| Phase 3: gRPC Client         | 3-4 days | Phase 2               |
| Phase 4: API Endpoints       | 5-6 days | Phase 3               |
| Phase 5: Testing             | 4-5 days | Phase 4               |
| Phase 6: Deployment          | 4-5 days | Phase 5               |
| Phase 7: Performance         | 3-4 days | Phase 6               |

**Total Duration**: 24-31 days (4-5 weeks)

---

## Next Steps

1. **Immediate**: Set up NestJS project with free tier optimizations
2. **Week 1**: Implement core infrastructure and authentication
3. **Week 2-3**: Develop API endpoints and gRPC integration
4. **Week 4**: Deploy to EC2 and optimize performance
5. **Month 6-10**: Monitor and prepare for migration
6. **Month 12**: Execute migration to paid services (ECS Fargate)
